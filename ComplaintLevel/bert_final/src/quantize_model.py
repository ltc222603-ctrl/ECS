import os
import torch
import torch.nn as nn
from torch.ao.quantization import quantize_dynamic

import config
from bert_final.src.models.bert_model import BertModel


def get_file_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)


def main():
    device = torch.device("cpu")
    print("device:", device)
    print("BEST_MODEL_PATH :", config.BEST_MODEL_PATH)
    print("QUANT_MODEL_PATH:", config.QUANT_MODEL_PATH)

    if not os.path.exists(config.BEST_MODEL_PATH):
        raise FileNotFoundError(f"原始模型不存在: {config.BEST_MODEL_PATH}")

    # 1) 加载 FP32 原模型
    model = BertModel().to(device)
    state_dict = torch.load(config.BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 2) 动态量化（主要量化 Linear 层）
    qmodel = quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8
    )
    qmodel.eval()

    # 3) 保存量化模型
    torch.save(qmodel, config.QUANT_MODEL_PATH)
    print("量化模型已保存:", config.QUANT_MODEL_PATH)

    # 4) 打印大小对比
    fp32_size = get_file_size_mb(config.BEST_MODEL_PATH)
    int8_size = get_file_size_mb(config.QUANT_MODEL_PATH)

    print(f"FP32 模型大小: {fp32_size:.2f} MB")
    print(f"INT8 模型大小: {int8_size:.2f} MB")
    if int8_size > 0:
        print(f"压缩比例: {fp32_size / int8_size:.2f}x")


if __name__ == "__main__":
    main()