import os
import random
import numpy as np
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from distill.src.dataset import DistillDataset
from bert_final.src.models.bert_model import BertModel
from distill.src.models.distill_bert_model import DistillBertStudent


def set_seed(seed):
    """
    固定随机种子，保证实验结果尽量可复现。
    包括 Python、NumPy、PyTorch 以及 GPU 的随机性。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """
    获取当前运行设备。
    如果有可用 GPU，则返回 cuda，否则返回 cpu。
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_tokenizers():
    """
    加载教师模型和学生模型各自对应的 tokenizer。
    教师模型和学生模型可能使用不同词表，因此需要分别加载。
    """
    teacher_tokenizer = AutoTokenizer.from_pretrained(config.TEACHER_PRETRAIN_DIR)
    student_tokenizer = AutoTokenizer.from_pretrained(config.STUDENT_PRETRAIN_DIR)
    return teacher_tokenizer, student_tokenizer


def build_dataloaders(teacher_tokenizer, student_tokenizer):
    """
    构建训练集、验证集、测试集对应的 DataLoader。

    同一条文本会分别经过教师 tokenizer 和学生 tokenizer 编码，
    因此这里使用自定义的 DistillDataset。
    """
    train_dataset = DistillDataset(
        file_path=config.TRAIN_PATH,
        teacher_tokenizer=teacher_tokenizer,
        student_tokenizer=student_tokenizer,
        label_map=config.LABEL_MAP,
        teacher_max_len=config.TEACHER_SEQ_LEN,
        student_max_len=config.STUDENT_SEQ_LEN,
    )

    val_dataset = DistillDataset(
        file_path=config.VAL_PATH,
        teacher_tokenizer=teacher_tokenizer,
        student_tokenizer=student_tokenizer,
        label_map=config.LABEL_MAP,
        teacher_max_len=config.TEACHER_SEQ_LEN,
        student_max_len=config.STUDENT_SEQ_LEN,
    )

    test_dataset = DistillDataset(
        file_path=config.TEST_PATH,
        teacher_tokenizer=teacher_tokenizer,
        student_tokenizer=student_tokenizer,
        label_map=config.LABEL_MAP,
        teacher_max_len=config.TEACHER_SEQ_LEN,
        student_max_len=config.STUDENT_SEQ_LEN,
    )

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.DISTILL_BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.DISTILL_BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.DISTILL_BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


def load_teacher(device):
    """
    加载教师模型。

    教师模型使用你已经训练好的 BERT 分类模型权重，
    并冻结其参数，使其在蒸馏过程中只负责提供软标签，不参与更新。
    """
    teacher = BertModel().to(device)
    teacher.load_state_dict(torch.load(config.BEST_MODEL_PATH, map_location=device))
    teacher.eval()

    # 冻结教师模型参数，防止训练时更新
    for param in teacher.parameters():
        param.requires_grad = False

    return teacher


def load_student(device):
    """
    加载学生模型。

    学生模型使用 distilbert-base-zh-cased 作为底座，
    通过 DistillBertStudent 封装成分类模型。
    """
    student = DistillBertStudent().to(device)
    return student


def build_optimizer(student_model):
    """
    构建学生模型优化器。

    对 bias 和 LayerNorm.weight 不使用 weight decay，
    其他参数使用配置文件中的 DISTILL_WEIGHT_DECAY。
    """
    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in student_model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": config.DISTILL_WEIGHT_DECAY,
        },
        {
            "params": [
                p for n, p in student_model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=config.DISTILL_LR)
    return optimizer


def build_scheduler(optimizer, train_loader):
    """
    构建学习率调度器。

    使用线性 warmup + 线性衰减的方式控制学习率变化，
    其中 warmup 步数由配置文件中的 DISTILL_WARMUP_RATIO 决定。
    """
    total_steps = len(train_loader) * config.DISTILL_EPOCHS
    warmup_steps = int(total_steps * config.DISTILL_WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    return scheduler


class EarlyStopping:
    """
    提前停止类。

    当验证集指标连续若干轮没有提升时，停止训练，
    防止后期过拟合，同时节省训练时间。
    """

    def __init__(self, patience=2):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def step(self, score):
        """
        输入当前验证集分数。
        如果分数提升，则重置计数器；
        如果未提升，则计数器加一；
        达到 patience 后触发 early stop。
        """
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
    """
    保存训练过程中的 loss 曲线和 accuracy 曲线。

    横坐标为 epoch，纵坐标分别为 loss 和 accuracy。
    用于后续实验分析和论文作图。
    """
    os.makedirs(save_dir, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss 曲线
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, history["train_loss"], marker="o", label="train_loss")
    plt.plot(epochs, history["val_loss"], marker="o", label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Distillation Loss Curve")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"distill_loss_curve_{timestamp}.png"))
    plt.close()

    # Accuracy 曲线
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, history["train_acc"], marker="o", label="train_acc")
    plt.plot(epochs, history["val_acc"], marker="o", label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Distillation Accuracy Curve")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"distill_acc_curve_{timestamp}.png"))
    plt.close()


def forward_teacher(teacher, batch, device):
    """
    教师模型前向传播。

    输入教师 tokenizer 编码后的 input_ids、attention_mask 等信息，
    输出教师模型的 logits，作为蒸馏中的软标签来源。
    """
    input_ids = batch["teacher_input_ids"].to(device, non_blocking=True)
    attention_mask = batch["teacher_attention_mask"].to(device, non_blocking=True)

    with torch.no_grad():
        if "teacher_token_type_ids" in batch:
            token_type_ids = batch["teacher_token_type_ids"].to(device, non_blocking=True)
            outputs = teacher(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
        else:
            outputs = teacher(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

    if isinstance(outputs, tuple):
        teacher_logits = outputs[-1]
    else:
        teacher_logits = outputs

    return teacher_logits


def forward_student(student, batch, device):
    """
    学生模型前向传播。

    输入学生 tokenizer 编码后的文本，
    输出学生模型的 logits。
    """
    input_ids = batch["student_input_ids"].to(device, non_blocking=True)
    attention_mask = batch["student_attention_mask"].to(device, non_blocking=True)

    logits = student(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    return logits


def distillation_loss(student_logits, teacher_logits, labels, temperature, alpha):
    """
    计算蒸馏总损失。

    总损失由两部分组成：
    1. CE Loss：学生模型对真实标签的分类损失
    2. KD Loss：学生模型输出分布与教师模型输出分布之间的蒸馏损失

    最终公式：
    Loss = alpha * KD_Loss + (1 - alpha) * CE_Loss
    """
    # 硬标签损失
    ce_loss = F.cross_entropy(student_logits, labels)

    # 软标签损失
    #取log(概率)*老师模型的每个概率
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

    # 让 student 的概率分布逼近 teacher 的概率分布
    kd_loss = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="batchmean",
    ) * (temperature ** 2)

    total_loss = alpha * kd_loss + (1.0 - alpha) * ce_loss
    return total_loss, ce_loss, kd_loss


def train_one_epoch(student, teacher, dataloader, optimizer, scheduler, scaler, device, epoch):
    """
    训练一个 epoch。

    每个 batch 中：
    1. 教师模型前向得到 soft logits
    2. 学生模型前向得到 student logits
    3. 计算蒸馏损失
    4. 反向传播并更新学生模型参数

    返回该 epoch 的平均 loss 和平均 accuracy。
    """
    student.train()
    teacher.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    pbar = tqdm(dataloader, desc=f"Train Epoch {epoch}", ncols=120)

    for batch in pbar:
        labels = batch["labels"].to(device, non_blocking=True)

        optimizer.zero_grad()

        teacher_logits = forward_teacher(teacher, batch, device)

        with torch.amp.autocast("cuda", enabled=(config.USE_AMP and device.type == "cuda")):
            student_logits = forward_student(student, batch, device)

            loss, ce_loss, kd_loss = distillation_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                labels=labels,
                temperature=config.DISTILL_TEMPERATURE,
                alpha=config.DISTILL_ALPHA,
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(student.parameters(), config.DISTILL_MAX_GRAD_NORM)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()

        preds = torch.argmax(student_logits, dim=1)
        total_correct += (preds == labels).sum().item()
        total_count += labels.size(0)

        current_acc = total_correct / total_count

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            ce=f"{ce_loss.item():.4f}",
            kd=f"{kd_loss.item():.4f}",
            acc=f"{current_acc:.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
        )

    avg_loss = total_loss / len(dataloader)
    avg_acc = total_correct / total_count
    return avg_loss, avg_acc


@torch.no_grad()
def evaluate(student, dataloader, device, mode="Val"):
    """
    在验证集或测试集上评估学生模型。

    计算：
    - 平均 loss
    - accuracy
    - precision
    - recall
    - f1

    这里使用加权平均指标，适合类别样本数可能不完全均衡的情况。
    """
    student.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    all_labels = []
    all_preds = []

    pbar = tqdm(dataloader, desc=mode, ncols=120)

    for batch in pbar:
        labels = batch["labels"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=(config.USE_AMP and device.type == "cuda")):
            logits = forward_student(student, batch, device)
            loss = F.cross_entropy(logits, labels)

        total_loss += loss.item()

        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == labels).sum().item()
        total_count += labels.size(0)

        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{total_correct / total_count:.4f}",
        )

    avg_loss = total_loss / len(dataloader)
    avg_acc = total_correct / total_count
    precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    return avg_loss, avg_acc, precision, recall, f1


def train():
    """
    蒸馏训练主函数。

    完整流程包括：
    1. 固定随机种子
    2. 加载设备、tokenizer、数据集
    3. 加载教师模型和学生模型
    4. 构建优化器和学习率调度器
    5. 进行蒸馏训练与验证
    6. 保存最优学生模型
    7. 保存训练曲线图片
    8. 在测试集上评估最优学生模型
    """
    set_seed(config.SEED)

    device = get_device()
    print("device:", device)
    print("teacher tokenizer:", config.TEACHER_PRETRAIN_DIR)
    print("student tokenizer:", config.STUDENT_PRETRAIN_DIR)
    print("teacher weight:", config.BEST_MODEL_PATH)
    print("distill save path:", config.DISTILL_MODEL_PATH)
    print("distill image path:", config.DISTILL_IMAGE_PATH)

    teacher_tokenizer, student_tokenizer = load_tokenizers()
    train_loader, val_loader, test_loader = build_dataloaders(
        teacher_tokenizer, student_tokenizer
    )

    teacher = load_teacher(device)
    student = load_student(device)

    optimizer = build_optimizer(student)
    scheduler = build_scheduler(optimizer, train_loader)
    scaler = torch.amp.GradScaler("cuda", enabled=(config.USE_AMP and device.type == "cuda"))
    early_stopping = EarlyStopping(patience=config.DISTILL_PATIENCE)

    best_val_acc = 0.0
    os.makedirs(os.path.dirname(config.DISTILL_MODEL_PATH), exist_ok=True)
    os.makedirs(config.DISTILL_IMAGE_PATH, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    save_model_path = config.DISTILL_MODEL_PATH.replace(".pt", f"_{timestamp}.pt")

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(config.DISTILL_EPOCHS):
        train_loss, train_acc = train_one_epoch(
            student=student,
            teacher=teacher,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            epoch=epoch + 1,
        )

        val_loss, val_acc, val_precision, val_recall, val_f1 = evaluate(
            student=student,
            dataloader=val_loader,
            device=device,
            mode=f"Val Epoch {epoch + 1}",
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch [{epoch + 1}/{config.DISTILL_EPOCHS}]")
        print(f"train loss: {train_loss:.4f}, train acc: {train_acc:.4f}")
        print(
            f"val   loss: {val_loss:.4f}, "
            f"val   acc: {val_acc:.4f}, "
            f"val   precision: {val_precision:.4f}, "
            f"val   recall: {val_recall:.4f}, "
            f"val   f1: {val_f1:.4f}"
        )

        # 如果当前验证集精度更高，则保存最优学生模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(student.state_dict(), save_model_path)
            print("best distill model saved:", save_model_path)

        improved = early_stopping.step(val_acc)
        if not improved:
            print(
                f"val acc 没提升，early stop counter: "
                f"{early_stopping.counter}/{config.DISTILL_PATIENCE}"
            )

        if early_stopping.early_stop:
            print("early stopping triggered")
            break

    print("distill train finished")
    print("best val acc:", best_val_acc)

    # 保存训练曲线
    save_curves(history, config.DISTILL_IMAGE_PATH, timestamp)

    # 加载验证集上表现最好的学生模型，再在测试集上评估
    student.load_state_dict(torch.load(save_model_path, map_location=device))

    test_loss, test_acc, test_precision, test_recall, test_f1 = evaluate(
        student=student,
        dataloader=test_loader,
        device=device,
        mode="Test",
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