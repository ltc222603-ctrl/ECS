"""图片查询 Action。

大致流程：
1. FastAPI 把图片 dataUrl、图片信息和用户问题放到 metadata 后，直接调用这个 action。
2. action 先通过 LangChain 调用视觉大模型，识别图片中的商品实体，并生成判断依据 reason。
3. 把识别出来的实体和 reason 拼成检索问题，复用 information_retrieval.py 里的 GraphRAG 查询数据库。
4. 如果数据库有结果，就把数据库结果和 reason 一起交给大模型生成回复。
5. 如果数据库没查到，就按“此图片查询的数据不在数据库中...”的提示生成兜底回复。
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

import dotenv
import requests
import yaml
from langchain.prompts import ChatPromptTemplate
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI
from rasa.utils.endpoints import EndpointConfig
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

BASE_DIR = Path(__file__).resolve().parent.parent

# action server 从 rasa 目录启动时，直接把 rasa 目录加进路径，方便复用 addons。
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from addons.information_retrieval import GraphRAG

logger = logging.getLogger(__name__)

DEFAULT_IMAGE_QUESTION = "请识别图片中的商品并查询相关信息"

_RETRIEVAL: GraphRAG | None = None
_CHAT_MODEL: ChatOpenAI | None = None


class OpenAICompatibleEmbeddings(Embeddings):
    """调用本地 embedding 服务，接口格式兼容 OpenAI。"""

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
    def name(self) -> str:
        return "action_image_product_query"

    async def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        _domain: dict[str, Any],
    ) -> list[dict[str, Any]]:
        metadata = tracker.latest_message.get("metadata") or {}
        image_payload = metadata.get("image") or {}
        image_data_url = image_payload.get("dataUrl")
        user_text = (metadata.get("user_text") or "").strip()
        user_text = user_text or (tracker.latest_message.get("text") or "").strip()
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

        # 先让视觉模型抽取实体和判断依据，后面再用这些信息查询图数据库。
        try:
            image_analysis = _extract_image_entities(image_data_url, user_text)
        except Exception as exc:
            logger.exception("Image entity extraction failed: %s", exc)
            dispatcher.utter_message(text=f"图片解析失败：{_compact_text(str(exc))}")
            return []

        reason = image_analysis.get("reason") or "图片可见信息不足，无法给出明确判断依据。"
        entities = image_analysis.get("entities") or []
        logger.info("图片解析实体: %s", entities)
        logger.info("图片判断依据: %s", reason)
        retrieval_query = _build_retrieval_query(user_text, image_analysis)
        logger.info("图生文 GraphRAG 查询: %s", retrieval_query)

        # 这里复用原来的 GraphRAG，不重新写数据库连接逻辑。
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
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                [
                    {"type": "text", "text": "{user_prompt}"},
                    {"type": "image_url", "image_url": {"url": "{image_data_url}"}},
                ],
            ),
        ]
    )
    content = _chat_completion(
        prompt,
        {
            "user_prompt": user_prompt,
            "image_data_url": image_data_url,
        },
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

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            (
                "human",
                "用户问题：{user_text}\n\n"
                "抽取实体：{entities}\n\n"
                "上下文：\n{context}\n\n"
                "请生成最终回复。",
            ),
        ]
    )
    return _chat_completion(
        prompt,
        {
            "user_text": user_text,
            "entities": json.dumps(entities, ensure_ascii=False),
            "context": context,
        },
    ).strip()


def _get_retrieval() -> GraphRAG:
    global _RETRIEVAL

    # GraphRAG 初始化较慢，缓存起来给后续图片请求复用。
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
    prompt: ChatPromptTemplate,
    variables: dict[str, Any] | None = None,
    response_format: dict[str, str] | None = None,
) -> str:
    langchain_messages = prompt.format_messages(**(variables or {}))
    chat_model = _get_chat_model()

    try:
        if response_format:
            chat_model = chat_model.bind(response_format=response_format)
        response = chat_model.invoke(langchain_messages)
    except Exception as exc:
        if not response_format:
            raise RuntimeError(f"视觉模型调用失败: {_compact_text(str(exc))}") from exc

        # 有些兼容接口不支持 response_format，失败后按普通对话再调用一次。
        try:
            response = _get_chat_model().invoke(langchain_messages)
        except Exception as retry_exc:
            raise RuntimeError(f"视觉模型调用失败: {_compact_text(str(retry_exc))}") from retry_exc

    return _message_content_to_text(response.content)


def _get_chat_model() -> ChatOpenAI:
    global _CHAT_MODEL

    if _CHAT_MODEL is not None:
        return _CHAT_MODEL

    dotenv.load_dotenv(BASE_DIR / ".env")
    api_key = os.getenv("OPENAI_API_KEY2") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL2") or os.getenv("OPENAI_BASE_URL")
    model = os.getenv("OPENAI_MODEL2") or os.getenv("OPENAI_MODEL")

    if not api_key or not base_url or not model:
        raise RuntimeError(
            "视觉模型配置未配置完整，请检查 OPENAI_API_KEY2、OPENAI_BASE_URL2、OPENAI_MODEL2，"
            "或 OPENAI_API_KEY、OPENAI_BASE_URL、OPENAI_MODEL。"
        )

    _CHAT_MODEL = ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=_clean_chat_base_url(base_url),
        temperature=0,
    )
    return _CHAT_MODEL


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict):
                texts.append(str(item.get("text", "")))
            else:
                texts.append(str(item))
        return "".join(texts)

    return str(content)


def _extract_json_object(content: str) -> dict[str, Any]:
    text = content.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError(f"模型未返回 JSON 对象: {content}")

    return json.loads(match.group())


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


def _clean_chat_base_url(base_url: str) -> str:
    return re.sub(r"/chat/completions/?$", "", base_url.rstrip("/"))


def _headers(api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _compact_text(text: str, max_length: int = 500) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."
