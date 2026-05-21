import sys
import os
import jieba
import numpy as np
import pandas as pd

# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(BASE_DIR)

import config

def load_data(file_path):
    # 1️⃣ 整行读取 txt.每列名称raw
    df = pd.read_csv(file_path, header=None, names=["raw"], encoding="utf-8")

    # 2️⃣ 从右边切分：最后一个是标签
    df[["text", "label"]] = df["raw"].str.rsplit(n=1, expand=True)

    # 3️⃣ 删除原始列
    df = df.drop(columns=["raw"])

    return df

def show_basic_info(df, name):
    # 前5条
    print("\n前5条数据：")
    print(df.head())

    # 类别统计
    print("\n类别统计：")
    print(df["label"].value_counts())

# 句子长度分布
def analyze_text_length(train_df, val_df, test_df):
    # 合并三个数据集
    all_df = pd.concat([train_df, val_df, test_df], axis=0, ignore_index=True)

    # 给每一条文本算“长度”，并新增一列叫 length
    all_df["length"] = all_df["text"].apply(len)

    # 计算均值和方差
    max_len = int(all_df["length"].quantile(0.95))

    print(f"\n整体文本长度分位数：{max_len:.2f}")

# 结巴分词函数ef process_text(text):
def process_text(text):
    words = jieba.lcut(text)
    return " ".join(words[:config.SEQ_LEN])


if __name__ == "__main__":
    # =====================
    # 读取数据
    # =====================
    train_df = load_data(config.TRAIN_PATH)
    val_df = load_data(config.VAL_PATH)
    test_df = load_data(config.TEST_PATH)

    # # 基础查看
    # show_basic_info(train_df, "训练集")
    # show_basic_info(val_df, "验证集")
    # show_basic_info(test_df, "测试集")
#--------------------------------------------------------
    # 句子长度分布
    analyze_text_length(train_df, val_df, test_df)
# --------------------------------------------------------
    # 将句子长度变为同一长度
    train_df["words"] = train_df["text"].apply(process_text)
    val_df["words"] = val_df["text"].apply(process_text)
    test_df["words"] = test_df["text"].apply(process_text)

    # 标签转数字
    train_df["label"] = train_df["label"].map(config.LABEL_MAP)
    val_df["label"] = val_df["label"].map(config.LABEL_MAP)
    test_df["label"] = test_df["label"].map(config.LABEL_MAP)

    # 保存
    train_df.to_csv(config.TRAIN_SPLIT_PATH, index=False, encoding="utf-8")
    val_df.to_csv(config.VAL_SPLIT_PATH, index=False, encoding="utf-8")
    test_df.to_csv(config.TEST_SPLIT_PATH, index=False, encoding="utf-8")





