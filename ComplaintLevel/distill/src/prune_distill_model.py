import os
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

import config
from distill.src.models.distill_bert_model import DistillBertStudent


def get_device():
    """
    获取设备。
    剪枝本身对设备要求不高，能用 GPU 就用 GPU，否则用 CPU。
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_file_size_mb(path):
    """
    计算文件大小（MB）。
    """
    return os.path.getsize(path) / (1024 * 1024)


def count_linear_sparsity(model):
    """
    统计模型中所有 Linear 层的稀疏率。
    稀疏率 = 0 的权重数量 / 总权重数量
    """
    zero_count = 0
    total_count = 0

    for module in model.modules():
        if isinstance(module, nn.Linear):
            weight = module.weight.data
            zero_count += torch.sum(weight == 0).item()
            total_count += weight.numel()

    sparsity = zero_count / total_count if total_count > 0 else 0.0
    return zero_count, total_count, sparsity


def main():
    device = get_device()
    print("device:", device)
    print("DISTILL_MODEL_PATH:", config.BEST_DISTILL_MODEL_PATH)
    print("PRUNED_DISTILL_MODEL_PATH:", config.PRUNED_DISTILL_MODEL_PATH)
    print("PRUNE_AMOUNT:", config.PRUNE_AMOUNT)

    if not os.path.exists(config.BEST_DISTILL_MODEL_PATH):
        raise FileNotFoundError(f"蒸馏模型不存在: {config.BEST_DISTILL_MODEL_PATH}")

    # 1) 加载蒸馏后的学生模型
    model = DistillBertStudent().to(device)
    model.load_state_dict(torch.load(config.BEST_DISTILL_MODEL_PATH, map_location=device))
    model.eval()

    # 剪枝前统计
    zero_before, total_before, sparsity_before = count_linear_sparsity(model)
    print(f"剪枝前 Linear 层稀疏率: {sparsity_before:.4f} ({zero_before}/{total_before})")

    # 2) 收集所有 Linear 层的 weight 做全局非结构化剪枝
    parameters_to_prune = []
    for module in model.modules():
        if isinstance(module, nn.Linear):
            parameters_to_prune.append((module, "weight"))

    if len(parameters_to_prune) == 0:
        raise ValueError("模型中没有找到可剪枝的 Linear 层。")

    # 3) 全局 L1 非结构化剪枝
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=config.PRUNE_AMOUNT,
    )

    # 4) 将 mask 应用到真实权重，并移除重参数化包装
    for module, name in parameters_to_prune:
        prune.remove(module, name)

    # 剪枝后统计
    zero_after, total_after, sparsity_after = count_linear_sparsity(model)
    print(f"剪枝后 Linear 层稀疏率: {sparsity_after:.4f} ({zero_after}/{total_after})")

    # 5) 保存剪枝后的模型
    torch.save(model.state_dict(), config.PRUNED_DISTILL_MODEL_PATH)
    print("剪枝模型已保存:", config.PRUNED_DISTILL_MODEL_PATH)

    # 6) 打印大小对比
    origin_size = get_file_size_mb(config.BEST_DISTILL_MODEL_PATH)
    pruned_size = get_file_size_mb(config.PRUNED_DISTILL_MODEL_PATH)

    print(f"原蒸馏模型大小: {origin_size:.2f} MB")
    print(f"剪枝后模型大小: {pruned_size:.2f} MB")
    if origin_size > 0:
        print(f"大小变化比例: {pruned_size / origin_size:.4f}")


if __name__ == "__main__":
    main()