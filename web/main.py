from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import requests
from .schemas import ChatRequest
from fastapi.staticfiles import StaticFiles

app = FastAPI()

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

# ===== 响应接口 =====
@app.post("/api/chat")
def chat(req: ChatRequest):
    """
    前端 ➜ FastAPI ➜ Rasa Pro
    """
    payload = {
        "sender": req.userId,     # userId 对应 Rasa sender_id
        "message": req.message
    }

    try:
        rasa_resp = requests.post(RASA_REST_URL, json=payload, timeout=300)
        rasa_resp.raise_for_status()
    except Exception as e:
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


#把 web/static 目录，当成一个网站，挂载在 http://localhost:8000/
app.mount(
    "/",
    StaticFiles(directory="web/static", html=True),
    name="static"
)