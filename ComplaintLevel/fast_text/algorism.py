import os
import time
import fasttext
import config

def train_fasttext(train_data_path,dev_data_path,test_data_path,number: str):
    model = fasttext.train_supervised(
        input=train_data_path,
        autotuneValidationFile=dev_data_path,
        autotuneDuration=180,#自动调参时间
        wordNgrams=2,
        verbose=1
    )

    N, precision, recall = model.test(test_data_path)

    print(f"样本数: {N}")
    print(f"Accuracy: {precision:.4f}")
    print(f"precision: {precision:.4f}")
    print(f"recall: {recall:.4f}")

    # name, ext = os.path.splitext(config.FASTTEXT_MODEL_PATH)
    # new_path = name + number + ext
    #
    # model.save_model(new_path)
    #
    # print("模型已保存：", new_path)

if __name__ == "__main__":
    # 以字方式分词训练
    # train_fasttext(config.FASTTEXT_SINGLE_TRAIN_PATH,config.FASTTEXT_SINGLE_VAL_PATH,
    #                config.FASTTEXT_SINGLE_TEST_PATH,01)

    # 以词方式分词训练
    train_fasttext(config.FASTTEXT_WORDS_TRAIN_PATH,config.FASTTEXT_WORDS_VAL_PATH,
                   config.FASTTEXT_WORDS_TEST_PATH,"02")