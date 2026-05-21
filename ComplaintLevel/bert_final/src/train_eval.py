import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from datetime import datetime
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

import config
from bert_final.src.dataset import MyDataset
from bert_final.src.models.bert_model import BertModel


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_dataloader():
    train_dataset = MyDataset(config.TRAIN_PATH)
    val_dataset = MyDataset(config.VAL_PATH)
    test_dataset = MyDataset(config.TEST_PATH)

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader


def build_optimizer(model):
    # 这些参数通常不做 weight_decay
    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": config.WEIGHT_DECAY
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0
        }
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=config.LR)
    return optimizer


def build_scheduler(optimizer, train_loader):
    total_steps = len(train_loader) * config.EPOCHS
    warmup_steps = int(total_steps * config.WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    return scheduler


class EarlyStopping:
    def __init__(self, patience=2):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    # 如果验证集指标连续几轮都没有提升，就提前结束训练
    def step(self, score):
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False


def save_curves(history, save_dir, timestamp):
    os.makedirs(save_dir, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss 曲线
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, history["train_loss"], marker="o", label="train_loss")
    plt.plot(epochs, history["val_loss"], marker="o", label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"loss_curve_{timestamp}.png"))
    plt.close()

    # Accuracy 曲线
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, history["train_acc"], marker="o", label="train_acc")
    plt.plot(epochs, history["val_acc"], marker="o", label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"acc_curve_{timestamp}.png"))
    plt.close()


def train_one_epoch(model, dataloader, optimizer, scheduler, scaler, device, epoch):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    pbar = tqdm(dataloader, desc=f"Train Epoch {epoch}", ncols=120)

    for batch in pbar:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=(config.USE_AMP and device.type == "cuda")):
            if "token_type_ids" in batch:
                token_type_ids = batch["token_type_ids"].to(device, non_blocking=True)
                loss, logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=labels
                )
            else:
                loss, logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()

        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == labels).sum().item()
        total_count += labels.size(0)

        current_acc = total_correct / total_count

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{current_acc:.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}"
        )

    avg_loss = total_loss / len(dataloader)
    avg_acc = total_correct / total_count
    return avg_loss, avg_acc


@torch.no_grad()
def evaluate(model, dataloader, device, mode="Val"):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    all_labels = []
    all_preds = []

    pbar = tqdm(dataloader, desc=mode, ncols=120)

    for batch in pbar:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=(config.USE_AMP and device.type == "cuda")):
            if "token_type_ids" in batch:
                token_type_ids = batch["token_type_ids"].to(device, non_blocking=True)
                loss, logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=labels
                )
            else:
                loss, logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

        total_loss += loss.item()

        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == labels).sum().item()
        total_count += labels.size(0)

        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{total_correct / total_count:.4f}"
        )

    avg_loss = total_loss / len(dataloader)
    avg_acc = total_correct / total_count

    precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    return avg_loss, avg_acc, precision, recall, f1


def train():
    set_seed(config.SEED)

    device = get_device()
    print("device:", device)

    train_loader, val_loader, test_loader = build_dataloader()

    model = BertModel().to(device)
    optimizer = build_optimizer(model)
    scheduler = build_scheduler(optimizer, train_loader)
    scaler = torch.amp.GradScaler("cuda", enabled=(config.USE_AMP and device.type == "cuda"))
    early_stopping = EarlyStopping(patience=config.PATIENCE)

    best_val_acc = 0.0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    save_path = f"{config.BERT_MODEL_PATH}_{timestamp}.pt"

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs(config.BERT_IMAGE_PATH, exist_ok=True)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    for epoch in range(config.EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            epoch=epoch + 1
        )

        val_loss, val_acc, val_precision, val_recall, val_f1 = evaluate(
            model=model,
            dataloader=val_loader,
            device=device,
            mode=f"Val Epoch {epoch + 1}"
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch [{epoch + 1}/{config.EPOCHS}]")
        print(f"train loss: {train_loss:.4f}, train acc: {train_acc:.4f}")
        print(
            f"val   loss: {val_loss:.4f}, "
            f"val   acc: {val_acc:.4f}, "
            f"val   precision: {val_precision:.4f}, "
            f"val   recall: {val_recall:.4f}, "
            f"val   f1: {val_f1:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print("best model saved:", save_path)

        improved = early_stopping.step(val_acc)
        if not improved:
            print(f"val acc 没提升，early stop counter: {early_stopping.counter}/{config.PATIENCE}")

        if early_stopping.early_stop:
            print("early stopping triggered")
            break

    print("train finished")
    print("best val acc:", best_val_acc)

    # 保存图片
    save_curves(history, config.BERT_IMAGE_PATH, timestamp)

    # 加载最优模型
    model.load_state_dict(torch.load(save_path, map_location=device))

    test_loss, test_acc, test_precision, test_recall, test_f1 = evaluate(
        model=model,
        dataloader=test_loader,
        device=device,
        mode="Test"
    )

    print(
        f"test  loss: {test_loss:.4f}, "
        f"test  acc: {test_acc:.4f}, "
        f"test  precision: {test_precision:.4f}, "
        f"test  recall: {test_recall:.4f}, "
        f"test  f1: {test_f1:.4f}"
    )


if __name__ == "__main__":
    train()