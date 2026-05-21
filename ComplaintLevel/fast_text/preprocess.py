import config
import os
import jieba

#分成单个字符
def to_char_level(text):
    text = str(text).replace(" ", "").strip()
    #list(text)自动按字符拆分
    return " ".join(list(text))


def convert_to_fasttext_char(input_path, output_path):
    # 先确保目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line_num, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                text, label = line.rsplit(maxsplit=1)
            except ValueError:
                print(f"第 {line_num} 行格式错误：{line}")
                continue

            # print(f"__label__{label} {to_char_level(text)}\n")
            # break
            fout.write(f"__label__{label} {to_char_level(text)}\n")


# jieba 分词
def to_jieba_level(text):
    text = str(text).strip()
    # 用 jieba 切词，再用空格拼接，符合 fastText 输入格式
    return " ".join(jieba.cut(text))


def convert_to_fasttext_jieba(input_path, output_path):
    # 先确保目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line_num, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                text, label = line.rsplit(maxsplit=1)
            except ValueError:
                print(f"第 {line_num} 行格式错误：{line}")
                continue

            fout.write(f"__label__{label} {to_jieba_level(text)}\n")

if __name__ == "__main__":
    # n-gram分字
    # convert_to_fasttext_char(config.TRAIN_PATH, config.FASTTEXT_SINGLE_TRAIN_PATH)
    # convert_to_fasttext_char(config.VAL_PATH, config.FASTTEXT_SINGLE_VAL_PATH)
    # convert_to_fasttext_char(config.TEST_PATH, config.FASTTEXT_SINGLE_TEST_PATH)

    # jieba 分词
    convert_to_fasttext_jieba(config.TRAIN_PATH, config.FASTTEXT_WORDS_TRAIN_PATH)
    convert_to_fasttext_jieba(config.VAL_PATH, config.FASTTEXT_WORDS_VAL_PATH)
    convert_to_fasttext_jieba(config.TEST_PATH, config.FASTTEXT_WORDS_TEST_PATH)