import time
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer

import config
from distill.src.models.distill_bert_model import DistillBertStudent


# =====================
# 请求与响应数据结构
# =====================
class PredictRequest(BaseModel):
    """
    输入请求体
    """
    text: str


class PredictResponse(BaseModel):
    """
    输出响应体
    """
    text: str
    pred_id: int
    pred_label: str
    pred_prob: float
    all_probs: dict
    elapsed_time: float
    model_path: str
    device: str


# =====================
# 物流投诉等级分类服务
# =====================
class LogisticsComplaintLevelService:
    """
    物流投诉等级分类服务类

    功能：
    1. 加载学生 tokenizer
    2. 加载剪枝后的蒸馏模型
    3. 提供单条物流投诉文本等级分类能力
    """

    def __init__(self):
        self.device = self.get_device()
        self.tokenizer = None
        self.model = None
        self.id2label = {v: k for k, v in config.LABEL_MAP.items()}

    def get_device(self):
        """
        获取推理设备。

        剪枝模型部署时更推荐使用 CPU，
        因为更符合轻量化部署场景，也更稳定。
        """
        return torch.device("cpu")

    def load(self):
        """
        加载 tokenizer 和剪枝后的蒸馏模型。
        服务启动时只执行一次。
        """
        self.tokenizer = AutoTokenizer.from_pretrained(config.STUDENT_PRETRAIN_DIR)

        self.model = DistillBertStudent().to(self.device)
        self.model.load_state_dict(
            torch.load(config.PRUNED_DISTILL_MODEL_PATH, map_location=self.device)
        )
        self.model.eval()

    @torch.no_grad()
    def predict(self, text: str):
        """
        对输入文本进行物流投诉等级分类预测。

        返回内容包括：
        - 预测类别 id
        - 预测标签
        - 最大概率
        - 全部类别概率分布
        - 推理耗时
        """
        if not text or not text.strip():
            raise ValueError("输入文本不能为空")

        tokens = self.tokenizer(
            text,
            max_length=config.STUDENT_SEQ_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = tokens["input_ids"].to(self.device)
        attention_mask = tokens["attention_mask"].to(self.device)

        start_time = time.perf_counter()

        logits = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        probs = torch.softmax(logits, dim=1)[0].cpu().tolist()
        pred_id = int(torch.argmax(logits, dim=1).item())
        pred_label = self.id2label[pred_id]
        pred_prob = float(probs[pred_id])

        end_time = time.perf_counter()
        elapsed_time = float(end_time - start_time)

        all_probs = {
            self.id2label[i]: float(probs[i])
            for i in range(len(probs))
        }

        return {
            "text": text,
            "pred_id": pred_id,
            "pred_label": pred_label,
            "pred_prob": pred_prob,
            "all_probs": all_probs,
            "elapsed_time": elapsed_time,
            "model_path": config.PRUNED_DISTILL_MODEL_PATH,
            "device": str(self.device),
        }


# =====================
# FastAPI 应用
# =====================
app = FastAPI(
    title="Logistics Complaint Level Classification Service",
    description="基于剪枝后蒸馏模型的物流投诉等级分类服务",
    version="1.0.0"
)

service = LogisticsComplaintLevelService()


@app.on_event("startup")
def startup_event():
    """
    服务启动时加载模型。
    """
    service.load()
    print("模型已加载完成")
    print("模型路径：", config.PRUNED_DISTILL_MODEL_PATH)
    print("设备：", service.device)


@app.get("/health")
def health():
    """
    健康检查接口。
    用于确认服务是否正常启动。
    """
    return {
        "status": "ok",
        "task": "logistics complaint level classification",
        "model_loaded": service.model is not None,
        "model_path": config.PRUNED_DISTILL_MODEL_PATH,
        "device": str(service.device)
    }


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """
    单条物流投诉文本等级分类接口。

    输入示例：
    {
        "text": "物流怎么不更新了"
    }

    输出说明：
    - pred_id: 预测类别编号
    - pred_label: 预测等级标签（low / medium / high）
    - pred_prob: 当前预测类别概率
    - all_probs: 三个类别的完整概率分布
    - elapsed_time: 本次推理耗时
    """
    try:
        result = service.predict(request.text)
        return PredictResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")


# =====================
# 本地运行入口
# =====================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "logistics_complain_bert:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )