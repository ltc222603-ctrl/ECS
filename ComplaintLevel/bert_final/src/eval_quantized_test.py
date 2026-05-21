import os
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import config
from bert_final.src.dataset import MyDataset
from bert_final.src.models.bert_model import BertModel


def get_device():
    # 为了公平比较，FP32 和 INT8 都固定在 CPU 上测试
    return torch.device("cpu")


def build_test_loader():
    test_dataset = MyDataset(config.TEST_PATH)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=False
    )
    return test_loader


def load_fp32_model(device):
    """
    加载原始 FP32 模型
    BEST_MODEL_PATH 保存的是 state_dict
    """
    if not os.path.exists(config.BEST_MODEL_PATH):
        raise FileNotFoundError(f"原始模型不存在: {config.BEST_MODEL_PATH}")

    model = BertModel().to(device)
    state_dict = torch.load(config.BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_int8_model(device):
    """
    加载量化后的 INT8 模型
    QUANT_MODEL_PATH 保存的是整个量化模型对象
    """
    if not os.path.exists(config.QUANT_MODEL_PATH):
        raise FileNotFoundError(f"量化模型不存在: {config.QUANT_MODEL_PATH}")

    try:
        model = torch.load(config.QUANT_MODEL_PATH, map_location=device, weights_only=False)
    except TypeError:
        model = torch.load(config.QUANT_MODEL_PATH, map_location=device)

    model.eval()
    return model


@torch.no_grad()
def evaluate(model, dataloader, device, model_name="MODEL"):
    model.eval()

    all_labels = []
    all_preds = []

    total_samples = 0
    total_batches = 0
    total_infer_time = 0.0

    pbar = tqdm(dataloader, desc=f"Evaluate {model_name}", ncols=120)

    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        start_time = time.perf_counter()

        if "token_type_ids" in batch:
            token_type_ids = batch["token_type_ids"].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
        else:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        end_time = time.perf_counter()

        total_infer_time += (end_time - start_time)
        total_batches += 1

        # 兼容不同 forward 返回格式
        # 可能是 logits，也可能是 (loss, logits) 或其他 tuple
        if isinstance(outputs, tuple):
            logits = outputs[-1]
        else:
            logits = outputs

        preds = torch.argmax(logits, dim=1)

        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())
        total_samples += labels.size(0)

        current_acc = accuracy_score(all_labels, all_preds)
        pbar.set_postfix(acc=f"{current_acc:.4f}")

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    avg_time_per_sample = total_infer_time / total_samples if total_samples > 0 else 0.0
    avg_time_per_batch = total_infer_time / total_batches if total_batches > 0 else 0.0
    throughput = total_samples / total_infer_time if total_infer_time > 0 else 0.0

    return {
        "samples": total_samples,
        "batches": total_batches,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total_infer_time": total_infer_time,
        "avg_time_per_sample": avg_time_per_sample,
        "avg_time_per_batch": avg_time_per_batch,
        "throughput": throughput
    }


def print_metrics(title, metrics):
    print(f"\n{'=' * 20} {title} {'=' * 20}")
    print(f"样本数:            {metrics['samples']}")
    print(f"批次数:            {metrics['batches']}")
    print(f"Accuracy:          {metrics['accuracy']:.4f}")
    print(f"Precision:         {metrics['precision']:.4f}")
    print(f"Recall:            {metrics['recall']:.4f}")
    print(f"F1:                {metrics['f1']:.4f}")
    print(f"总推理耗时:        {metrics['total_infer_time']:.4f} 秒")
    print(f"单 batch 平均耗时: {metrics['avg_time_per_batch']:.6f} 秒")
    print(f"单样本平均耗时:    {metrics['avg_time_per_sample']:.6f} 秒")
    print(f"吞吐量:            {metrics['throughput']:.2f} samples/sec")


def print_compare(fp32_metrics, int8_metrics):
    print(f"\n{'=' * 20} 对比结果 {'=' * 20}")
    print(f"Accuracy  : FP32 = {fp32_metrics['accuracy']:.4f} | INT8 = {int8_metrics['accuracy']:.4f}")
    print(f"Precision : FP32 = {fp32_metrics['precision']:.4f} | INT8 = {int8_metrics['precision']:.4f}")
    print(f"Recall    : FP32 = {fp32_metrics['recall']:.4f} | INT8 = {int8_metrics['recall']:.4f}")
    print(f"F1        : FP32 = {fp32_metrics['f1']:.4f} | INT8 = {int8_metrics['f1']:.4f}")
    print(f"样本耗时   : FP32 = {fp32_metrics['avg_time_per_sample']:.6f}s | INT8 = {int8_metrics['avg_time_per_sample']:.6f}s")
    print(f"吞吐量     : FP32 = {fp32_metrics['throughput']:.2f} | INT8 = {int8_metrics['throughput']:.2f} samples/sec")

    acc_diff = int8_metrics["accuracy"] - fp32_metrics["accuracy"]
    f1_diff = int8_metrics["f1"] - fp32_metrics["f1"]
    speedup = (
        fp32_metrics["avg_time_per_sample"] / int8_metrics["avg_time_per_sample"]
        if int8_metrics["avg_time_per_sample"] > 0 else 0.0
    )

    print(f"\nAccuracy 差值(INT8 - FP32): {acc_diff:+.4f}")
    print(f"F1 差值(INT8 - FP32):       {f1_diff:+.4f}")
    print(f"单样本推理加速比:            {speedup:.2f}x")


def main():
    device = get_device()
    print("device:", device)
    print("BEST_MODEL_PATH :", config.BEST_MODEL_PATH)
    print("QUANT_MODEL_PATH:", config.QUANT_MODEL_PATH)
    print("TEST_PATH       :", config.TEST_PATH)
    print("BATCH_SIZE      :", config.BATCH_SIZE)

    if not os.path.exists(config.TEST_PATH):
        raise FileNotFoundError(f"测试集不存在: {config.TEST_PATH}")

    test_loader = build_test_loader()

    # FP32
    fp32_model = load_fp32_model(device)
    fp32_metrics = evaluate(fp32_model, test_loader, device, model_name="FP32")
    print_metrics("FP32 Result", fp32_metrics)

    # INT8
    int8_model = load_int8_model(device)
    int8_metrics = evaluate(int8_model, test_loader, device, model_name="INT8")
    print_metrics("INT8 Result", int8_metrics)

    # Compare
    print_compare(fp32_metrics, int8_metrics)


if __name__ == "__main__":
    main()