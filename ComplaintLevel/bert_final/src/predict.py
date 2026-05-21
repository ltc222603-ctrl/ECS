import time
import torch
from transformers import AutoTokenizer

import config
from bert_final.src.models.bert_model import BertModel


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PRETRAIN_DIR)

    model = BertModel().to(device)
    model.load_state_dict(torch.load(config.BEST_MODEL_PATH, map_location=device))
    model.eval()

    return tokenizer, model, device


def predict_one(text, tokenizer, model, device):
    tokens = tokenizer(
        text,
        max_length=config.SEQ_LEN,
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
        if "token_type_ids" in tokens:
            token_type_ids = tokens["token_type_ids"].to(device)
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
        else:
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

    if device.type == "cuda":
        torch.cuda.synchronize()

    end_time = time.perf_counter()

    pred_id = torch.argmax(logits, dim=1).item()
    id2label = {v: k for k, v in config.LABEL_MAP.items()}
    pred_label = id2label[pred_id]

    elapsed_time = end_time - start_time

    return pred_id, pred_label, elapsed_time


if __name__ == "__main__":
    tokenizer, model, device = load_model()

    while True:
        #系统里写着在运输中，不过已经好几天没变化了 0
        #联系配送员比较困难，沟通不太顺畅 1
        #外观上有轻微问题，感觉不像是全新的 2
        text = input("请输入文本：").strip()
        if not text:
            break

        pred_id, pred_label, elapsed_time = predict_one(text, tokenizer, model, device)

        print("文本：", text)
        print("预测类别id：", pred_id)
        print("预测类别：", pred_label)
        print(f"预测耗时：{elapsed_time:.6f} 秒")
        print("-" * 40)