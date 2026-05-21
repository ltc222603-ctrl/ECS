import os

# 当前文件的路径的上一级文件夹（就是 ComplaintLevel）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据目录
DATA_DIR = os.path.join(BASE_DIR, "data")

# 数据路径
TRAIN_PATH = os.path.join(DATA_DIR, "train.txt")
VAL_PATH = os.path.join(DATA_DIR, "val.txt")
TEST_PATH = os.path.join(DATA_DIR, "test.txt")


# 机器学习数据目录
FOREST_DIR = os.path.join(DATA_DIR, "forest_data")
TRAIN_SPLIT_PATH = os.path.join(FOREST_DIR, "train_split.txt")
VAL_SPLIT_PATH = os.path.join(FOREST_DIR, "val_split.txt")
TEST_SPLIT_PATH = os.path.join(FOREST_DIR, "test_split.txt")
STOPWORDS_HIT_PATH = os.path.join(FOREST_DIR, "stopwords_cn.txt")

#fast text数据目录
FASTTEXT_DIR = os.path.join(DATA_DIR, "fasttext_data")
FASTTEXT_SINGLE_TRAIN_PATH = os.path.join(FASTTEXT_DIR, "single_train_fasttext.txt")
FASTTEXT_SINGLE_VAL_PATH = os.path.join(FASTTEXT_DIR, "single_val_fasttext.txt")
FASTTEXT_SINGLE_TEST_PATH = os.path.join(FASTTEXT_DIR, "single_test_fasttext.txt")

FASTTEXT_WORDS_TRAIN_PATH = os.path.join(FASTTEXT_DIR, "words_train_fasttext.txt")
FASTTEXT_WORDS_VAL_PATH = os.path.join(FASTTEXT_DIR, "words_val_fasttext.txt")
FASTTEXT_WORDS_TEST_PATH = os.path.join(FASTTEXT_DIR, "words_test_fasttext.txt")



# 模型目录
MODEL_DIR = os.path.join(BASE_DIR, "model")

# 预训练bert模型目录
MODEL_PRETRAIN_DIR = os.path.join(
    MODEL_DIR,
    "model_pretrain",
    "bert",
    "bert-base-chinese"
)

# 预训练bert蒸馏模型目录
STUDENT_PRETRAIN_DIR = os.path.join(
    MODEL_DIR,
    "model_pretrain",
    "bert",
    "distilbert-base-zh-cased"
)


#保存随机森林模型路径
FOREST_MODEL_PATH = os.path.join(MODEL_DIR, "forest_model.pkl")
VECTORIZER_PATH=os.path.join(MODEL_DIR, "vectorizer_model.pkl")

#保存fasttext模型路径
FASTTEXT_MODEL_PATH = os.path.join(MODEL_DIR, "fasttext_model.pkl")

#保存bert模型路径+加时间
BERT_MODEL_PATH = os.path.join(MODEL_DIR, "bert_model")

#保存bert模型训练图片路径
BERT_IMAGE_PATH = os.path.join(MODEL_DIR, "bert_image")

#最好的预训练bert模型
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "bert_model_20260510_1745.pt")

#量化bert模型保存路径
QUANT_MODEL_PATH = os.path.join(MODEL_DIR, "qbert_model_int8.pt")

#模型蒸馏保存路径
DISTILL_MODEL_PATH = os.path.join(MODEL_DIR, "distill_bert_model.pt")

#保存蒸馏模型训练图片保存路径
DISTILL_IMAGE_PATH = os.path.join(MODEL_DIR, "distill_image")

#保存最好的蒸馏模型
BEST_DISTILL_MODEL_PATH = os.path.join(MODEL_DIR, "distill_bert_model_20260510_1721.pt")

# 剪枝后蒸馏模型保存路径
PRUNED_DISTILL_MODEL_PATH = os.path.join(MODEL_DIR, "pruned_distill_bert_model.pt")


#数据类别映射
LABEL_MAP = {"low": 0, "medium": 1, "high": 2}

# =====================
# 模型参数
# =====================
SEQ_LEN = 56


# =====================
# 模型参数（随机森林）
# =====================
RANDOM_STATE = 42
N_ESTIMATORS = 100
MAX_DEPTH = None
MIN_SAMPLES_SPLIT = 2
MIN_SAMPLES_LEAF = 1

# =====================
# 模型参数（bert）
# =====================
SEED = 42
BATCH_SIZE = 32
DROPOUT = 0.3
LR = 1e-5
EPOCHS = 2
NUM_WORKERS = 2
WEIGHT_DECAY = 0.02
WARMUP_RATIO = 0.10
MAX_GRAD_NORM = 1.0
PATIENCE = 1
USE_AMP = True

# 老师模型预训练目录
TEACHER_PRETRAIN_DIR = MODEL_PRETRAIN_DIR

# 老师和学生各自的最大长度
TEACHER_SEQ_LEN = SEQ_LEN
STUDENT_SEQ_LEN = SEQ_LEN

# # =====================
# # 模型参数（distill）
# # =====================
#
# # 老师模型预训练目录
# TEACHER_PRETRAIN_DIR = MODEL_PRETRAIN_DIR
#
# # 老师和学生各自的最大长度
# TEACHER_SEQ_LEN = SEQ_LEN
# STUDENT_SEQ_LEN = SEQ_LEN
#
# # 蒸馏训练参数
# DISTILL_BATCH_SIZE = BATCH_SIZE
# DISTILL_LR = 1e-5
# DISTILL_EPOCHS = 5
# DISTILL_WEIGHT_DECAY = 0.01
# DISTILL_WARMUP_RATIO = 0.1
# DISTILL_MAX_GRAD_NORM = 1.0
# DISTILL_PATIENCE = 2
# # 蒸馏超参数
# DISTILL_TEMPERATURE = 2.0
# DISTILL_ALPHA = 0.5
#
# # 学生模型 dropout
# STUDENT_DROPOUT = 0.2




# # =====================
# # 模型参数（distill）
# # =====================
#
# DISTILL_BATCH_SIZE = BATCH_SIZE
#
# # 学生模型容易过拟合，学习率稍微降一点
# DISTILL_LR = 8e-6
#
# DISTILL_EPOCHS = 5
# DISTILL_WEIGHT_DECAY = 0.02
# DISTILL_WARMUP_RATIO = 0.1
# DISTILL_MAX_GRAD_NORM = 1.0
# DISTILL_PATIENCE = 2
#
# # 让学生更多学习 teacher 的软标签
# DISTILL_TEMPERATURE = 2.5
# DISTILL_ALPHA = 0.65
#
# # 提高 dropout，缓解训练集很快 100% 的问题
# STUDENT_DROPOUT = 0.3



# =====================
# 模型参数（distill）
# =====================

DISTILL_BATCH_SIZE = BATCH_SIZE
DISTILL_LR = 1e-5
DISTILL_EPOCHS = 4
DISTILL_WEIGHT_DECAY = 0.02
DISTILL_WARMUP_RATIO = 0.1
DISTILL_MAX_GRAD_NORM = 1.0
DISTILL_PATIENCE = 2

DISTILL_TEMPERATURE = 2.0
DISTILL_ALPHA = 0.3

STUDENT_DROPOUT = 0.3

# 剪枝比例，例如 0.2 表示剪掉 20% 的权重
PRUNE_AMOUNT = 0.2