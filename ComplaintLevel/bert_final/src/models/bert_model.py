import torch
import torch.nn as nn
import config
from torch.utils.data import DataLoader
from transformers import AutoModel
from bert_final.src.dataset import MyDataset

class BertModel(nn.Module):
    def __init__(self):
        super().__init__()

        # 加载本地预训练模型
        self.bert = AutoModel.from_pretrained(config.MODEL_PRETRAIN_DIR)

        # dropout
        self.dropout = nn.Dropout(config.DROPOUT)

        # 分类层
        self.fc = nn.Linear(self.bert.config.hidden_size, len(config.LABEL_MAP))

        # 损失函数
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        cls_output = outputs.pooler_output
        x = self.dropout(cls_output)
        logits = self.fc(x)

        if labels is not None:
            loss = self.loss_fn(logits, labels)
            return loss, logits

        return logits

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = MyDataset(config.TRAIN_PATH)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    model = BertModel().to(device)
    model.eval()

    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        if "token_type_ids" in batch:
            token_type_ids = batch["token_type_ids"].to(device)
            loss, logits = model(input_ids, attention_mask, token_type_ids, labels)
        else:
            loss, logits = model(input_ids, attention_mask, labels=labels)

        print("input_ids shape:", input_ids.shape)
        print("attention_mask shape:", attention_mask.shape)
        print("labels shape:", labels.shape)
        print("logits shape:", logits.shape)
        print("loss:", loss.item())
        break