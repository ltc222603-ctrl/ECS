import torch
import torch.nn as nn
import config
from transformers import AutoModel


class DistillBertStudent(nn.Module):
    def __init__(self):
        super().__init__()

        # 加载学生预训练模型
        self.distil_bert = AutoModel.from_pretrained(config.STUDENT_PRETRAIN_DIR)

        # dropout
        self.dropout = nn.Dropout(config.DROPOUT)

        # 分类层
        self.fc = nn.Linear(self.distil_bert.config.hidden_size, len(config.LABEL_MAP))

        # 损失函数
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.distil_bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # DistilBERT 没有 pooler_output，取 [CLS] 位置
        cls_output = outputs.last_hidden_state[:, 0]

        x = self.dropout(cls_output)
        logits = self.fc(x)

        if labels is not None:
            loss = self.loss_fn(logits, labels)
            return loss, logits

        return logits