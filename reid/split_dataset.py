import os
import shutil
import random

def make_file(file_path: str):
    if os.path.exists(file_path):
        shutil.rmtree(file_path)
    os.makedirs(file_path)

def is_image(filename):
    img_ext = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
    basename = os.path.basename(filename)
    basename_ext = os.path.splitext(basename)[-1]
    return (basename_ext in img_ext) and (not basename.startswith("."))


def scan_image_dir(image_dir):
    all_images = os.listdir(image_dir)
    all_images = [x for x in all_images if is_image(x)]
    return all_images


def write_lines_to_file(filename, lines, path, index):
    with open(filename, 'a') as f:
        x = path + '/' + lines.strip() + ' ' + index + "\n"
        f.write(os.path.join(x))
        print(x)
    print("done write to: ", filename)


split_rate = 0.4
data_path = r'/home/brl4090/pzm/Person_reID_baseline_pytorch/Redpanda/rpd17'
output_path = r'/home/brl4090/pzm/Person_reID_baseline_pytorch/Redpanda/fold_4'
fold = ['val.txt', 'gallery.txt', 'query.txt', 'train.txt']

data_class = [cla for cla in os.listdir(data_path)]
print("数据的种类分别为：")
print(data_class)

# 建立保存训练集的文件夹
train_root = os.path.join(output_path, "train")  # 训练集的文件夹名称为 train
make_file(train_root)
for num_class in data_class:
    # 建立每个类别对应的文件夹
    make_file(os.path.join(train_root, num_class))

# 建立保存验证集的文件夹
val_root = os.path.join(output_path, "val")  # 验证集的文件夹名称为 val
make_file(val_root)
for num_class in data_class:
    # 建立每个类别对应的文件夹
    make_file(os.path.join(val_root, num_class))

# 建立保存测试集的文件夹
gallery_root = os.path.join(output_path, "gallery")  # 测试集的文件夹名称为 gallery
make_file(gallery_root)
for num_class in data_class:
    # 建立每个类别对应的文件夹
    make_file(os.path.join(gallery_root, num_class))

# 建立保存测试集的文件夹
query_root = os.path.join(output_path, "query")  # 测试集的文件夹名称为 query
make_file(query_root)
for num_class in data_class:
    # 建立每个类别对应的文件夹
    make_file(os.path.join(query_root, num_class))

for num_class in data_class:
    num_class_path = os.path.join(data_path, num_class)
    images = scan_image_dir(num_class_path)
    num = len(images)

    rand = random.randint(0, 59)
    random.seed(rand)
    random.shuffle(images)
    else_index = random.sample(images, k=int(num * split_rate))
    else_num = len(else_index)

    rand = random.randint(0, 13)
    random.seed(rand)
    random.shuffle(else_index)
    test_index = random.sample(else_index, k=int(else_num * 0.5))
    test_num = len(test_index)
    other_half_index = list(set(else_index) - set(test_index))
    other_half_num = len(other_half_index)

    rand = random.randint(0, 71)
    random.seed(rand)
    random.shuffle(test_index)
    query_index = random.sample(test_index, k=int(test_num * 0.5))

    rand = random.randint(0, 93)
    random.seed(rand)
    random.shuffle(other_half_index)
    val_index = random.sample(other_half_index, k=int(other_half_num * 0.5))

    for index, image in enumerate(images):
        if image in query_index:
            data_image_path = os.path.join(num_class_path, image)
            query_new_path = os.path.join(query_root, num_class)
            shutil.copy(data_image_path, query_new_path)

        elif image in test_index:
            data_image_path = os.path.join(num_class_path, image)
            gallery_new_path = os.path.join(gallery_root, num_class)
            shutil.copy(data_image_path, gallery_new_path)

        elif image in val_index:
            data_image_path = os.path.join(num_class_path, image)
            val_new_path = os.path.join(val_root, num_class)
            shutil.copy(data_image_path, val_new_path)

        else:
            data_image_path = os.path.join(num_class_path, image)
            train_new_path = os.path.join(train_root, num_class)
            shutil.copy(data_image_path, train_new_path)
    print("\r[{}] split_rating [{}/{}]".format(num_class, index + 1, num), end="")  # processing bar
    print()

print("       ")
print("       ")
print("划分成功")
