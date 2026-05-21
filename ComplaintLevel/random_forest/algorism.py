import pandas as pd
import config
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold


# 读取数据
train_df = pd.read_csv(config.TRAIN_SPLIT_PATH)
val_df = pd.read_csv(config.VAL_SPLIT_PATH)
test_df = pd.read_csv(config.TEST_SPLIT_PATH)

# 合并 train 和 val，作为网格搜索与交叉验证的数据
train_val_df = pd.concat([train_df, val_df], axis=0, ignore_index=True)

# 文本和标签
X_train_val_text = train_val_df["words"]
y_train_val = train_val_df["label"]

X_test_text = test_df["words"]
y_test = test_df["label"]

# 读取停用词
with open(config.STOPWORDS_HIT_PATH, "r", encoding="utf-8") as f:
    stopwords = [line.strip() for line in f]

# 创建 TF-IDF 工具
vectorizer = TfidfVectorizer(
    stop_words=stopwords,
    tokenizer=str.split,
    preprocessor=None,
    token_pattern=None,
    lowercase=False
)

# 文本转向量
X_train_val = vectorizer.fit_transform(X_train_val_text)
X_test = vectorizer.transform(X_test_text)

rf = RandomForestClassifier(
    random_state=config.RANDOM_STATE,
    class_weight="balanced"
)

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [5, 8, 12],
    "min_samples_leaf": [5, 10, 20],         # 新增：控制叶子节点
    "min_samples_split": [10,20],       # 新增：控制分裂条件
}

# 交叉验证策略
cv = StratifiedKFold(
    n_splits=5,#数据分成5分，4份训练 + 1份验证
    shuffle=True,
    random_state=config.RANDOM_STATE
)

# 网格搜索 + 交叉验证
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring="f1_macro",   # 多分类推荐用 macro F1
    cv=cv,
    n_jobs=-1,#并行计算
    verbose=1#打印训练过程
)

# 训练
grid_search.fit(X_train_val, y_train_val)

# 输出最优参数和交叉验证结果
print("最优参数：")
print(grid_search.best_params_)

print("\n最佳交叉验证分数（F1-macro）：")
print(grid_search.best_score_)

# 拿到最优模型
best_model = grid_search.best_estimator_

# 创建模型保存目录（如果不存在）
os.makedirs(os.path.dirname(config.FOREST_MODEL_PATH), exist_ok=True)

#保存模型
joblib.dump(best_model, config.FOREST_MODEL_PATH)
joblib.dump(vectorizer, config.VECTORIZER_PATH)

print("\n模型已保存：")
print(config.FOREST_MODEL_PATH)
# print(config.VECTORIZER_PATH)

# 测试集预测
test_pred = best_model.predict(X_test)

print("\n测试集结果：")
print("Accuracy:", accuracy_score(y_test, test_pred))
print("Precision:", precision_score(y_test, test_pred, average="macro"))
print("Recall:", recall_score(y_test, test_pred, average="macro"))
print("F1:", f1_score(y_test, test_pred, average="macro"))

print("\n分类报告：")
print(classification_report(y_test, test_pred, digits=4))

