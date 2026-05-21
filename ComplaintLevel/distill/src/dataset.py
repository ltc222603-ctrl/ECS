import torch
from torch.utils.data import Dataset


class DistillDataset(Dataset):
    """
    同一条文本分别用 teacher tokenizer 和 student tokenizer 编码
    数据格式：句子 空格 标签
    """

    def __init__(
        self,
        file_path,
        teacher_tokenizer,
        student_tokenizer,
        label_map,
        teacher_max_len=85,
        student_max_len=85,
    ):
        self.file_path = file_path
        self.teacher_tokenizer = teacher_tokenizer
        self.student_tokenizer = student_tokenizer
        self.label_map = label_map
        self.teacher_max_len = teacher_max_len
        self.student_max_len = student_max_len
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

        teacher_enc = self.teacher_tokenizer(
            text,
            max_length=self.teacher_max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        student_enc = self.student_tokenizer(
            text,
            max_length=self.student_max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        item = {
            "teacher_input_ids": teacher_enc["input_ids"].squeeze(0),
            "teacher_attention_mask": teacher_enc["attention_mask"].squeeze(0),
            "student_input_ids": student_enc["input_ids"].squeeze(0),
            "student_attention_mask": student_enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(label_id, dtype=torch.long),
        }

        if "token_type_ids" in teacher_enc:
            item["teacher_token_type_ids"] = teacher_enc["token_type_ids"].squeeze(0)

        if "token_type_ids" in student_enc:
            item["student_token_type_ids"] = student_enc["token_type_ids"].squeeze(0)

        return item