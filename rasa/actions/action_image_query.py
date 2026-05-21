from __future__ import annotations

import json
import logging
import os
import re
import sys
import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Text, cast

import dotenv
import requests
import yaml
from langchain_core.embeddings import Embeddings
from rasa.utils.endpoints import EndpointConfig
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_DIR = BASE_DIR.parent
INFORMATION_RETRIEVAL_PATH = BASE_DIR / "addons" / "information_retrieval.py"

for path in (PROJECT_DIR, BASE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

spec = importlib.util.spec_from_file_location(
    "ecs_addons_information_retrieval",
    INFORMATION_RETRIEVAL_PATH,
)
if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load GraphRAG from {INFORMATION_RETRIEVAL_PATH}")

information_retrieval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(information_retrieval)
GraphRAG = cast(Any, information_retrieval).GraphRAG

logger = logging.getLogger(__name__)

IMAGE_QUERY_MARKER = "[image_product_query]"
DEFAULT_IMAGE_QUESTION = "请识别图片中的商品并查询相关信息"

_RETRIEVAL: GraphRAG | None = None


class ModelCallError(RuntimeError):
    def __init__(self, status_code: int, response_text: str) -> None:
        self.status_code = status_code
        self.response_text = response_text
        super().__init__(f"model api returned {status_code}: {response_text[:500]}")


class OpenAICompatibleEmbeddings(Embeddings):
    """Use the local embedding service with an OpenAI-compatible API."""

    def __init__(self, api_base: str, api_key: str | None, model: str) -> None:
        self.api_base = api_base
        self.api_key = api_key
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = requests.post(
            _join_api_path(self.api_base, "embeddings"),
            headers=_headers(self.api_key),
            json={"model": self.model, "input": texts},
            timeout=120,
        )
        response.raise_for_status()

        payload = response.json()
        items = sorted(payload.get("data", []), key=lambda item: item.get("index", 0))
        return [item["embedding"] for item in items]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class ActionImageProductQuery(Action):
    def name(self) -> Text:
        return "action_image_product_query"

    async def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        metadata = tracker.latest_message.get("metadata") or {}
        image_payload = metadata.get("image") or {}
        image_data_url = image_payload.get("dataUrl")
        user_text = (metadata.get("user_text") or "").strip()
        user_text = user_text or _strip_image_marker(tracker.latest_message.get("text"))
        user_text = user_text or DEFAULT_IMAGE_QUESTION

        logger.info(
            "图生文请求: sender=%s, slot_user_id=%s, user_text=%s, image_name=%s, image_type=%s, image_size=%s",
            tracker.sender_id,
            tracker.get_slot("user_id"),
            user_text,
            image_payload.get("name"),
            image_payload.get("type"),
            image_payload.get("size"),
        )

        if not image_data_url:
            dispatcher.utter_message(text="请先上传一张图片，再告诉我想查询什么。")
            return []

        try:
            image_analysis = _extract_image_entities(image_data_url, user_text)
        except Exception as exc:
            logger.exception("Image entity extraction failed: %s", exc)
            dispatcher.utter_message(text=_format_image_parse_error(exc))
            return []

        reason = image_analysis.get("reason") or "图片可见信息不足，无法给出明确判断依据。"
        entities = image_analysis.get("entities") or []
        logger.info("图片解析实体: %s", entities)
        logger.info("图片判断依据: %s", reason)
        retrieval_query = _build_retrieval_query(user_text, image_analysis)
        logger.info("图生文 GraphRAG 查询: %s", retrieval_query)

        try:
            search_results = await _get_retrieval().search(
                retrieval_query,
                _build_tracker_state(tracker, retrieval_query),
            )
            database_result = _search_results_to_text(search_results)
            logger.info("图生文数据库查询结果: %s", database_result or "空")
        except Exception as exc:
            logger.exception("Image database query failed: %s", exc)
            database_result = ""

        if database_result:
            answer_context = (
                "数据库查询结果如下：\n"
                f"{database_result}\n\n"
                f"图片判断依据：{reason}"
            )
            answer = _generate_final_answer(
                user_text=user_text,
                entities=entities,
                context=answer_context,
                database_hit=True,
            )
            logger.info("图生文最终回复类型: database_hit")
        else:
            fallback_context = f"此图片查询的数据不在数据库中，进行全网搜索的结果为：{reason}"
            answer = _generate_final_answer(
                user_text=user_text,
                entities=entities,
                context=fallback_context,
                database_hit=False,
            )
            logger.info("图生文最终回复类型: database_empty")

        logger.info("图生文最终回复: %s", answer)
        dispatcher.utter_message(text=answer)
        return []


def _extract_image_entities(image_data_url: str, user_text: str) -> dict[str, Any]:
    system_prompt = (
        "你是电商商品图片识别和检索实体抽取助手。"
        "请根据图片和用户问题抽取后续 text-to-sql / 图数据库查询需要的实体。"
        "只能输出 JSON，不要输出 Markdown。"
        "JSON 字段："
        "entities: 数组，每项包含 label、entity、confidence；"
        "label 只能是 Category1、Category2、Category3、Trademark、SPU、SKU、Attr；"
        "query: 用于数据库查询的一句话；"
        "reason: 中文判断依据，必须说明图片中可见证据和不确定性。"
        "不要臆测看不见的信息；不确定具体型号时，抽取更宽泛的品牌、品类、属性。"
    )
    user_prompt = (
        f"用户问题：{user_text}\n"
        "请识别图片中的商品、品牌、品类、型号、颜色、关键属性等。"
        "reason 示例：图片中可见 Apple 标志和竖向双摄，但没有显示包装盒、型号文字或系统页面，无法完全确认具体型号。"
    )

    content = _chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            },
        ],
        response_format={"type": "json_object"},
    )
    try:
        payload = _extract_json_object(content)
    except ValueError:
        logger.warning("Vision model returned non-json content: %s", content)
        payload = {
            "entities": [],
            "query": user_text,
            "reason": content.strip(),
        }

    return _normalize_image_analysis(payload)


def _generate_final_answer(
    user_text: str,
    entities: list[dict[str, Any]],
    context: str,
    database_hit: bool,
) -> str:
    if database_hit:
        system_prompt = (
            "你是电商智能客服。请基于数据库查询结果和图片判断依据回答用户。"
            "不要编造数据库结果之外的商品、价格、库存或订单信息。"
            "回复要自然、简洁，并说明识别依据。"
        )
    else:
        system_prompt = (
            "你是电商智能客服。数据库没有查到匹配数据时，"
            "请基于给定的图片判断依据谨慎回复，必须明确提示数据库无匹配。"
            "不要声称已经真实联网检索到了额外证据。"
        )

    user_prompt = (
        f"用户问题：{user_text}\n\n"
        f"抽取实体：{json.dumps(entities, ensure_ascii=False)}\n\n"
        f"上下文：\n{context}\n\n"
        "请生成最终回复。"
    )

    return _chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    ).strip()


def _get_retrieval() -> GraphRAG:
    global _RETRIEVAL

    if _RETRIEVAL is not None:
        return _RETRIEVAL

    dotenv.load_dotenv(BASE_DIR / ".env")
    endpoints = _read_yaml(BASE_DIR / "endpoints.yml")
    vector_store_config = endpoints.get("vector_store", {})
    embedding_config = _get_embedding_config(endpoints)

    embeddings = OpenAICompatibleEmbeddings(
        api_base=embedding_config["api_base"],
        api_key=embedding_config.get("api_key"),
        model=embedding_config["model"],
    )
    retrieval = GraphRAG(embeddings)
    retrieval.connect(EndpointConfig(**vector_store_config))

    _RETRIEVAL = retrieval
    return retrieval


def _get_embedding_config(endpoints: dict[str, Any]) -> dict[str, str]:
    for group in endpoints.get("model_groups", []):
        if group.get("id") != "embedding_models":
            continue

        models = group.get("models") or []
        if models:
            config = {
                key: _resolve_env_value(value)
                for key, value in models[0].items()
                if value is not None
            }
            return {
                "api_base": config.get("api_base", "http://localhost:10010"),
                "api_key": config.get("api_key", ""),
                "model": config.get("model", "bge-base-zh-v1.5"),
            }

    return {
        "api_base": "http://localhost:10010",
        "api_key": os.getenv("OPENAI_API_KEY", ""),
        "model": "bge-base-zh-v1.5",
    }


def _build_retrieval_query(user_text: str, image_analysis: dict[str, Any]) -> str:
    entities = image_analysis.get("entities") or []
    entity_text = "，".join(
        f"{item.get('label')}={item.get('entity')}"
        for item in entities
        if item.get("entity")
    )
    query = image_analysis.get("query") or user_text
    reason = image_analysis.get("reason") or ""

    return (
        f"用户问题：{user_text}\n"
        f"图片识别查询：{query}\n"
        f"图片抽取实体：{entity_text}\n"
        f"图片判断依据：{reason}\n"
        "请查询数据库中匹配的商品、SKU、品牌、品类或属性信息。"
    )


def _build_tracker_state(tracker: Tracker, query: str) -> dict[str, Any]:
    slots = dict(tracker.current_slot_values())

    return {
        "slots": slots,
        "events": [
            {
                "event": "user",
                "text": query,
            }
        ],
    }


def _search_results_to_text(search_results: Any) -> str:
    results = getattr(search_results, "results", None) or []
    texts: list[str] = []

    for result in results:
        text = (getattr(result, "text", "") or "").strip()
        if text and text not in {"空", "[]", "{}"}:
            texts.append(text)

    return "\n".join(texts[:10])


def _normalize_image_analysis(payload: dict[str, Any]) -> dict[str, Any]:
    entities = []

    for item in payload.get("entities", []) or []:
        if not isinstance(item, dict):
            continue

        label = str(item.get("label", "")).strip()
        entity = str(item.get("entity", "")).strip()
        if not label or not entity:
            continue

        entities.append(
            {
                "label": label,
                "entity": entity,
                "confidence": item.get("confidence"),
            }
        )

    return {
        "entities": entities,
        "query": str(payload.get("query") or "").strip(),
        "reason": str(payload.get("reason") or "").strip(),
    }


def _chat_completion(
    messages: list[dict[str, Any]],
    response_format: dict[str, str] | None = None,
) -> str:
    dotenv.load_dotenv(BASE_DIR / ".env")

    api_key = os.getenv("OPENAI_API_KEY2") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL2") or os.getenv("OPENAI_BASE_URL")
    model = os.getenv("OPENAI_MODEL2") or os.getenv("OPENAI_MODEL")

    if not api_key or not base_url or not model:
        raise RuntimeError(
            "视觉模型配置未配置完整，请检查 OPENAI_API_KEY2、OPENAI_BASE_URL2、OPENAI_MODEL2，"
            "或 OPENAI_API_KEY、OPENAI_BASE_URL、OPENAI_MODEL。"
        )

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0,
    }

    if response_format:
        payload["response_format"] = response_format

    response = requests.post(
        _join_api_path(base_url, "chat/completions"),
        headers=_headers(api_key),
        json=payload,
        timeout=180,
    )

    if response.status_code >= 400 and response_format:
        payload.pop("response_format", None)
        response = requests.post(
            _join_api_path(base_url, "chat/completions"),
            headers=_headers(api_key),
            json=payload,
            timeout=180,
        )

    if response.status_code >= 400:
        raise ModelCallError(response.status_code, response.text)

    try:
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception as exc:
        raise ModelCallError(response.status_code, response.text) from exc


def _extract_json_object(content: str) -> dict[str, Any]:
    text = content.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError(f"模型未返回 JSON 对象: {content}")

    return json.loads(match.group())


def _strip_image_marker(text: str | None) -> str:
    text = (text or "").replace(IMAGE_QUERY_MARKER, "").strip()
    return re.sub(r"^用户上传了一张商品图片。用户问题：", "", text).strip()


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def _resolve_env_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value

    return re.sub(
        r"\$\{([^}]+)\}",
        lambda match: os.getenv(match.group(1), match.group(0)),
        value,
    )


def _join_api_path(base_url: str, path: str) -> str:
    clean_base = base_url.rstrip("/")
    clean_path = path.strip("/")

    if clean_base.endswith(clean_path):
        return clean_base

    return f"{clean_base}/{clean_path}"


def _headers(api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _format_image_parse_error(exc: Exception) -> str:
    if isinstance(exc, ModelCallError):
        response_text = _compact_text(exc.response_text)
        return (
            f"图片解析失败：视觉模型接口返回 {exc.status_code}。\n"
            "请检查视觉模型的 API Key、Base URL、模型名称是否可用，并确认该模型支持图片输入。\n"
            f"接口返回：{response_text}"
        )

    return f"图片解析失败：{_compact_text(str(exc))}"


def _compact_text(text: str, max_length: int = 500) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."
