import os
import shutil
import random


def make_file(file_path: str):
    if os.path.exists(file_path):
        shutil.rmtree(file_path)
    os.makedirs(file_path)


split_rate = 0.45  # 将数据集中?%的数据划分到验证集中
img_path = r'/home/moon/yy/reid-animal/database/meta-iPanda-50/iPanda-10'  # 数据集存放的地方，建议在程序所在的文件夹下新建一个data文件夹，将需要划分的数据集存放进去
output_root = r'/home/moon/yy/reid-animal/database/iPanda-test'  # 这里是生成的训练集、验证集和测试集所处的位置，这里设置的是在当前文件夹下。

data_class = [cla for cla in os.listdir(img_path)]
print("数据的种类分别为：")
print(data_class)

# 建立保存训练集的文件夹
train_path = os.path.join(output_root, "gallery")  # 训练集的文件夹名称为 gallery
make_file(train_path)
for num_class in data_class:
    # 建立每个类别对应的文件夹
    make_file(os.path.join(train_path, num_class))

# 建立保存验证集的文件夹
val_path = os.path.join(output_root, "query")  # 验证集的文件夹名称为 query
make_file(val_path)
for num_class in data_class:
    # 建立每个类别对应的文件夹
    make_file(os.path.join(val_path, num_class))

for num_class in data_class:

    num_class_path = os.path.join(img_path, num_class)
    images = os.listdir(num_class_path)
    num = len(images)

    random.shuffle(images)
    else_index = random.sample(images, k=int(num * split_rate))

    for index, image in enumerate(images):
        if image in else_index:

            data_image_path = os.path.join(num_class_path, image)
            new_test_path = os.path.join(val_path, num_class)
            shutil.copy(data_image_path, new_test_path)

        else:

            data_image_path = os.path.join(num_class_path, image)
            new_train_path = os.path.join(train_path, num_class)
            shutil.copy(data_image_path, new_train_path)

    print("\r[{}] split_rating [{}/{}]".format(num_class, index + 1, num), end="")
    print()

print("split finished.")
