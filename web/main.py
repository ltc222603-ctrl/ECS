from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import requests
from .schemas import ChatRequest
from fastapi.staticfiles import StaticFiles

app = FastAPI()
logger = logging.getLogger("web")

# ===== CORS（前后端分离时必须）=====
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 开发阶段直接放开
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== Rasa Pro REST 接口地址 =====
RASA_REST_URL = "http://localhost:5005/webhooks/rest/webhook"
RASA_TRACKER_URL = "http://localhost:5005/conversations/{sender_id}/tracker"
RASA_ACTION_URL = "http://localhost:5055/webhook"
DEFAULT_IMAGE_MESSAGE = "请识别图片中的商品并查询相关信息"

# ===== 响应接口 =====
@app.post("/api/chat")
def chat(req: ChatRequest):
    """
    前端 ➜ FastAPI ➜ Rasa Pro
    """
    message = (req.message or "").strip()
    image_payload = None
    if req.image:
        image_payload = (
            req.image.model_dump(exclude_none=True)
            if hasattr(req.image, "model_dump")
            else req.image.dict(exclude_none=True)
        )
    if image_payload:
        # 图片消息直接走 action server，避免被 EnterpriseSearchPolicy 当成知识库搜索。
        logger.info(
            "收到图片消息: sender=%s, text=%s, image_name=%s, image_type=%s, image_size=%s",
            req.userId,
            message,
            image_payload.get("name"),
            image_payload.get("type"),
            image_payload.get("size"),
        )
        return call_image_action(req.userId, message, image_payload)

    payload = {
        "sender": req.userId,     # userId 对应 Rasa sender_id
        "message": message,
    }

    try:
        rasa_resp = requests.post(RASA_REST_URL, json=payload, timeout=300)
        rasa_resp.raise_for_status()
    except Exception:
        return {
            "message": "Rasa 服务暂时不可用，请稍后再试"
        }

    messages = rasa_resp.json()

    normalized = []
    for msg in messages:
        normalized.append({
            "text": msg.get("text"),
            "buttons": msg.get("buttons", []),
        })

    return {
        "messages": normalized
    }


def call_image_action(user_id: str, message: str, image_payload: dict):
    """
    图片消息直接调用 Rasa Action Server，避免被 EnterpriseSearchPolicy 当知识库搜索处理。
    """
    slots = get_rasa_tracker_slots(user_id)
    user_message = message or DEFAULT_IMAGE_MESSAGE
    logger.info("图片消息使用 tracker slots: sender=%s, user_id=%s", user_id, slots.get("user_id"))
    metadata = {
        "type": "image_product_query",
        "image": image_payload,
        "user_text": message,
    }
    action_payload = {
        "next_action": "action_image_product_query",
        "sender_id": user_id,
        "tracker": {
            "sender_id": user_id,
            "slots": slots,
            "latest_message": {
                "intent": {"name": "image_product_query_intent", "confidence": 1.0},
                "entities": [],
                "text": user_message,
                "metadata": metadata,
            },
            "events": [
                {
                    "event": "user",
                    "text": user_message,
                    "metadata": metadata,
                }
            ],
            "paused": False,
            "followup_action": None,
            "active_loop": {},
            "latest_action_name": "action_listen",
        },
        "domain": {
            "actions": ["action_image_product_query"],
            "responses": {},
            "slots": {},
            "entities": [],
            "intents": ["image_product_query_intent"],
        },
        "version": "3.1",
    }

    try:
        logger.info("调用图片 Action Server: action=action_image_product_query, sender=%s", user_id)
        action_resp = requests.post(RASA_ACTION_URL, json=action_payload, timeout=300)
        action_resp.raise_for_status()
    except Exception as exc:
        logger.exception("调用图片 Action Server 失败: %s", exc)
        return {
            "message": "图生文服务暂时不可用，请确认 Rasa Actions 已启动"
        }

    result = action_resp.json() or {}
    normalized = [
        {
            "text": msg.get("text"),
            "buttons": msg.get("buttons", []),
        }
        for msg in result.get("responses", [])
    ]
    logger.info("图片 Action Server 返回消息数: %s", len(normalized))

    return {
        "messages": normalized or [{"text": "图生文服务未返回结果", "buttons": []}]
    }


def get_rasa_tracker_slots(sender_id: str) -> dict:
    """
    读取 Rasa Core 当前会话槽位。图片 action 绕过 Core 时仍然要使用业务 user_id 槽位。
    """
    try:
        logger.info("读取 Rasa tracker slots: sender=%s", sender_id)
        tracker_resp = requests.get(
            RASA_TRACKER_URL.format(sender_id=sender_id),
            timeout=10,
        )
        tracker_resp.raise_for_status()
        tracker = tracker_resp.json() or {}
        slots = tracker.get("slots") or {}
        logger.info("读取 Rasa tracker slots 成功: sender=%s, user_id=%s", sender_id, slots.get("user_id"))
        return slots
    except Exception as exc:
        logger.warning("读取 Rasa tracker slots 失败: sender=%s, error=%s", sender_id, exc)
        return {}


#把 web/static 目录，当成一个网站，挂载在 http://localhost:8000/
app.mount(
    "/",
    StaticFiles(directory="web/static", html=True),
    name="static"
)
