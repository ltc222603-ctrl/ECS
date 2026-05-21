import os
import time
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoTokenizer

import config
from bert_final.src.models.bert_model import BertModel
from distill.src.models.distill_bert_model import DistillBertStudent


class EvalDataset(Dataset):
    """
    通用评估数据集
    用指定 tokenizer 对 test.txt 做编码
    数据格式：句子 空格 标签
    """

    def __init__(self, file_path, tokenizer, label_map, max_len):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.max_len = max_len
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []

        with open(self.file_path, "r", encoding="utf-8") as f:
            for line_num, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line:
                    continue

                try:
                    text, label = line.rsplit(" ", 1)
                except ValueError:
                    raise ValueError(
                        f"{self.file_path} 第 {line_num} 行格式错误，应为：句子 空格 标签；实际内容：{line}"
                    )

                if label not in self.label_map:
                    raise ValueError(
                        f"{self.file_path} 第 {line_num} 行标签非法：{label}"
                    )

                samples.append((text, self.label_map[label]))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label_id = self.samples[idx]

        enc = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(label_id, dtype=torch.long),
        }

        if "token_type_ids" in enc:
            item["token_type_ids"] = enc["token_type_ids"].squeeze(0)

        return item


def get_device():
    """
    为了公平比较，统一放在 CPU 上测试。
    """
    return torch.device("cpu")


def get_file_size_mb(path):
    """
    返回文件大小（MB）。
    """
    return os.path.getsize(path) / (1024 * 1024)


def build_test_loader(tokenizer_path):
    """
    根据指定 tokenizer 构建测试集 DataLoader
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    test_dataset = EvalDataset(
        file_path=config.TEST_PATH,
        tokenizer=tokenizer,
        label_map=config.LABEL_MAP,
        max_len=config.SEQ_LEN,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=False,
    )
    return test_loader


def load_bert_model(device):
    """
    加载原始最优 BERT 模型
    """
    if not os.path.exists(config.BEST_MODEL_PATH):
        raise FileNotFoundError(f"BERT 模型不存在: {config.BEST_MODEL_PATH}")

    model = BertModel().to(device)
    model.load_state_dict(torch.load(config.BEST_MODEL_PATH, map_location=device))
    model.eval()
    return model


def load_distill_model(model_path, device):
    """
    加载蒸馏学生模型 / 剪枝后蒸馏模型
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型不存在: {model_path}")

    model = DistillBertStudent().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


@torch.no_grad()
def evaluate_bert(model, dataloader, device, name="BERT"):
    """
    评估原始 BERT 模型
    """
    model.eval()

    all_labels = []
    all_preds = []

    total_samples = 0
    total_batches = 0
    total_infer_time = 0.0

    pbar = tqdm(dataloader, desc=f"Evaluate {name}", ncols=120)

    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        start_time = time.perf_counter()

        if "token_type_ids" in batch:
            token_type_ids = batch["token_type_ids"].to(device)
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
        else:
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        end_time = time.perf_counter()

        preds = torch.argmax(logits, dim=1)

        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

        total_samples += labels.size(0)
        total_batches += 1
        total_infer_time += (end_time - start_time)

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
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total_infer_time": total_infer_time,
        "avg_time_per_sample": avg_time_per_sample,
        "avg_time_per_batch": avg_time_per_batch,
        "throughput": throughput,
    }


@torch.no_grad()
def evaluate_student(model, dataloader, device, name="STUDENT"):
    """
    评估蒸馏学生模型 / 剪枝后学生模型
    """
    model.eval()

    all_labels = []
    all_preds = []

    total_samples = 0
    total_batches = 0
    total_infer_time = 0.0

    pbar = tqdm(dataloader, desc=f"Evaluate {name}", ncols=120)

    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        start_time = time.perf_counter()

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        end_time = time.perf_counter()

        preds = torch.argmax(logits, dim=1)

        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

        total_samples += labels.size(0)
        total_batches += 1
        total_infer_time += (end_time - start_time)

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
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total_infer_time": total_infer_time,
        "avg_time_per_sample": avg_time_per_sample,
        "avg_time_per_batch": avg_time_per_batch,
        "throughput": throughput,
    }


def print_metrics(title, metrics, model_path):
    """
    打印单个模型测试结果
    """
    print(f"\n{'=' * 20} {title} {'=' * 20}")
    print("模型路径:", model_path)
    print(f"模型大小:          {get_file_size_mb(model_path):.2f} MB")
    print(f"样本数:            {metrics['samples']}")
    print(f"Accuracy:          {metrics['accuracy']:.4f}")
    print(f"Precision:         {metrics['precision']:.4f}")
    print(f"Recall:            {metrics['recall']:.4f}")
    print(f"F1:                {metrics['f1']:.4f}")
    print(f"总推理耗时:        {metrics['total_infer_time']:.4f} 秒")
    print(f"单 batch 平均耗时: {metrics['avg_time_per_batch']:.6f} 秒")
    print(f"单样本平均耗时:    {metrics['avg_time_per_sample']:.6f} 秒")
    print(f"吞吐量:            {metrics['throughput']:.2f} samples/sec")


def print_compare(bert_metrics, distill_metrics, pruned_metrics):
    """
    打印三模型对比结果
    """
    print(f"\n{'=' * 20} 三模型对比 {'=' * 20}")
    print(f"Accuracy  : BERT = {bert_metrics['accuracy']:.4f} | DISTILL = {distill_metrics['accuracy']:.4f} | PRUNED = {pruned_metrics['accuracy']:.4f}")
    print(f"Precision : BERT = {bert_metrics['precision']:.4f} | DISTILL = {distill_metrics['precision']:.4f} | PRUNED = {pruned_metrics['precision']:.4f}")
    print(f"Recall    : BERT = {bert_metrics['recall']:.4f} | DISTILL = {distill_metrics['recall']:.4f} | PRUNED = {pruned_metrics['recall']:.4f}")
    print(f"F1        : BERT = {bert_metrics['f1']:.4f} | DISTILL = {distill_metrics['f1']:.4f} | PRUNED = {pruned_metrics['f1']:.4f}")

    print(f"\n样本耗时   : BERT = {bert_metrics['avg_time_per_sample']:.6f}s | DISTILL = {distill_metrics['avg_time_per_sample']:.6f}s | PRUNED = {pruned_metrics['avg_time_per_sample']:.6f}s")
    print(f"吞吐量     : BERT = {bert_metrics['throughput']:.2f} | DISTILL = {distill_metrics['throughput']:.2f} | PRUNED = {pruned_metrics['throughput']:.2f} samples/sec")

    print(f"\nDISTILL 相对 BERT 的 Accuracy 差值: {distill_metrics['accuracy'] - bert_metrics['accuracy']:+.4f}")
    print(f"PRUNED  相对 BERT 的 Accuracy 差值: {pruned_metrics['accuracy'] - bert_metrics['accuracy']:+.4f}")
    print(f"PRUNED  相对 DISTILL 的 Accuracy 差值: {pruned_metrics['accuracy'] - distill_metrics['accuracy']:+.4f}")


def main():
    device = get_device()
    print("device:", device)
    print("BEST_MODEL_PATH:", config.BEST_MODEL_PATH)
    print("BEST_DISTILL_MODEL_PATH:", config.BEST_DISTILL_MODEL_PATH)
    print("PRUNED_DISTILL_MODEL_PATH:", config.PRUNED_DISTILL_MODEL_PATH)
    print("TEST_PATH:", config.TEST_PATH)

    if not os.path.exists(config.TEST_PATH):
        raise FileNotFoundError(f"测试集不存在: {config.TEST_PATH}")

    # 分别构建 BERT 和学生模型的测试集
    bert_test_loader = build_test_loader(config.MODEL_PRETRAIN_DIR)
    student_test_loader = build_test_loader(config.STUDENT_PRETRAIN_DIR)

    # 1. 原始 BERT
    bert_model = load_bert_model(device)
    bert_metrics = evaluate_bert(bert_model, bert_test_loader, device, name="BERT")
    print_metrics("原始 BERT 模型结果", bert_metrics, config.BEST_MODEL_PATH)

    # 2. 蒸馏模型
    distill_model = load_distill_model(config.BEST_DISTILL_MODEL_PATH, device)
    distill_metrics = evaluate_student(distill_model, student_test_loader, device, name="DISTILL")
    print_metrics("蒸馏模型结果", distill_metrics, config.BEST_DISTILL_MODEL_PATH)

    # 3. 剪枝后蒸馏模型
    pruned_model = load_distill_model(config.PRUNED_DISTILL_MODEL_PATH, device)
    pruned_metrics = evaluate_student(pruned_model, student_test_loader, device, name="PRUNED")
    print_metrics("剪枝后蒸馏模型结果", pruned_metrics, config.PRUNED_DISTILL_MODEL_PATH)

    # 三模型对比
    print_compare(bert_metrics, distill_metrics, pruned_metrics)


if __name__ == "__main__":
    main()