import torch
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# ================= 设备 =================
device = "cuda" if torch.cuda.is_available() else "cpu"

# ================= 路径 =================
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

model_path = PROJECT_DIR / "pretrained" / "Qwen" / "Qwen2.5-3B-Instruct"

train_path = PROJECT_DIR / "output/4_train_test_split/ft_splits/train.jsonl"
val_path = PROJECT_DIR / "output/4_train_test_split/ft_splits/val.jsonl"

output_dir = BASE_DIR / "lora_output"
model_dir = BASE_DIR / "model" / "lora_model"

output_dir.mkdir(exist_ok=True, parents=True)
model_dir.mkdir(exist_ok=True, parents=True)

# ================= 基础参数 =================
max_seq_length = 1024
random_seed = 42
set_seed(random_seed)

# ================= tokenizer =================
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ================= 模型 =================
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    trust_remote_code=True,
)

model.config.use_cache = False

# ================= LoRA =================
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,  # 稍微调高
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, peft_config)

# ================= 加载数据 =================
dataset = load_dataset(
    "json",
    data_files={
        "train": str(train_path),
        "validation": str(val_path),
    }
)

train_dataset = dataset["train"]
val_dataset = dataset["validation"]

# ================= 数据格式化 =================
def formatting_func(example):
    messages = [
        {"role": "user", "content": example["prompt"].strip()},
        {"role": "assistant", "content": example["completion"].strip()},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    return {"text": text}

train_dataset = train_dataset.map(formatting_func, remove_columns=train_dataset.column_names)
val_dataset = val_dataset.map(formatting_func, remove_columns=val_dataset.column_names)

# 关键步骤：flatten
train_dataset = train_dataset.flatten()
val_dataset = val_dataset.flatten()

# ================= 训练参数 =================
training_args = TrainingArguments(
    output_dir=str(output_dir),
    per_device_train_batch_size=8,  # RTX 5070 可以 8
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    learning_rate=1e-4,  # 更稳定
    logging_steps=20,
    evaluation_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=500,
    warmup_ratio=0.05,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    fp16=torch.cuda.is_available(),
    report_to="none",
    seed=random_seed,
)

# ================= Trainer =================
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    args=training_args,
)

# ================= 开始训练 =================
trainer.train()

# ================= 保存 =================
model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)

print("训练完成，LoRA 已保存到：", model_dir)