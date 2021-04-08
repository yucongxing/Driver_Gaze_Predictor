import os
import random
import math
import shutil


def data_split(old_path):
    new_path = "./data"
    if os.path.exists("./data") == 0:
        os.makedirs(new_path)
    for root_dir, sub_dirs, file in os.walk(old_path):  # 遍历os.walk(）返回的每一个三元组，内容分别放在三个变量中
        for sub_dir in sub_dirs:
            file_names = os.listdir(os.path.join(root_dir, sub_dir))  # 遍历每个次级目录
            file_names = list(
                filter(lambda x: x.endswith(".jpg"), file_names)
            )  # 去掉列表中的非jpg格式的文件

            random.shuffle(file_names)
            for i in range(len(file_names)):
                if i < math.floor(0.8 * len(file_names)):
                    sub_path = os.path.join(new_path, "train_set", sub_dir)
                elif i < math.floor(0.9 * len(file_names)):
                    sub_path = os.path.join(new_path, "val_set", sub_dir)
                elif i < len(file_names):
                    sub_path = os.path.join(new_path, "test_set", sub_dir)
                if os.path.exists(sub_path) == 0:
                    os.makedirs(sub_path)

                shutil.copy(
                    os.path.join(root_dir, sub_dir, file_names[i]),
                    os.path.join(sub_path, file_names[i]),
                )  # 复制图片，从源到目的地


if __name__ == "__main__":
    data_path = "../../download/imagefiles"
    data_split(data_path)
