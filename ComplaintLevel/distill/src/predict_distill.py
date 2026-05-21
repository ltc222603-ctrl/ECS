import time
import torch
from transformers import AutoTokenizer

import config
from distill.src.models.distill_bert_model import DistillBertStudent


def get_device():
    """
    获取当前运行设备。
    有 GPU 用 GPU，否则用 CPU。
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    """
    加载学生 tokenizer 和蒸馏后的学生模型。
    """
    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(config.STUDENT_PRETRAIN_DIR)

    model = DistillBertStudent().to(device)
    model.load_state_dict(torch.load(config.BEST_DISTILL_MODEL_PATH, map_location=device))
    model.eval()

    return tokenizer, model, device


def predict_one(text, tokenizer, model, device):
    """
    对单条文本进行预测。
    返回：
    - pred_id: 预测类别 id
    - pred_label: 预测类别名称
    - pred_prob: 预测类别概率
    - elapsed_time: 推理耗时
    """
    tokens = tokenizer(
        text,
        max_length=config.STUDENT_SEQ_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    with torch.no_grad():
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

    probs = torch.softmax(logits, dim=1)
    pred_id = torch.argmax(probs, dim=1).item()
    pred_prob = probs[0][pred_id].item()

    if device.type == "cuda":
        torch.cuda.synchronize()

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    id2label = {v: k for k, v in config.LABEL_MAP.items()}
    pred_label = id2label[pred_id]

    return pred_id, pred_label, pred_prob, elapsed_time


if __name__ == "__main__":
    tokenizer, model, device = load_model()

    print("当前设备：", device)
    print("蒸馏模型路径：", config.BEST_DISTILL_MODEL_PATH)
    print("学生 tokenizer 路径：", config.STUDENT_PRETRAIN_DIR)
    print("-" * 50)

    while True:
        text = input("请输入文本：").strip()
        if not text:
            break

        pred_id, pred_label, pred_prob, elapsed_time = predict_one(
            text, tokenizer, model, device
        )

        print("文本：", text)
        print("预测类别id：", pred_id)
        print("预测类别：", pred_label)
        print(f"预测概率：{pred_prob:.4f}")
        print(f"预测耗时：{elapsed_time:.6f} 秒")
        print("-" * 50)