import config
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


class MyDataset(Dataset):
    def __init__(self, file_path, max_len=config.SEQ_LEN):
        self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PRETRAIN_DIR)
        self.max_len = max_len

        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        self.df = pd.DataFrame({"line": lines})
        self.df[["text", "label"]] = self.df["line"].str.rsplit(n=1, expand=True)
        self.df["label"] = self.df["label"].map(config.LABEL_MAP)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = row["text"]
        label = row["label"]

        tokens = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        item = {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }

        if "token_type_ids" in tokens:
            item["token_type_ids"] = tokens["token_type_ids"].squeeze(0)

        return item


if __name__ == "__main__":
    train_dataset = MyDataset(config.TRAIN_PATH)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    for batch in train_loader:
        print("input_ids shape:", batch["input_ids"].shape)
        print("attention_mask shape:", batch["attention_mask"].shape)
        print("labels shape:", batch["labels"].shape)

        # print(batch["input_ids"])
        # print(batch["attention_mask"])
        # print(batch["labels"])
        break