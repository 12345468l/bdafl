import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
def image():
    from PIL import Image
    import os
    import numpy as np

    # 设定数据集路径
    train_dir = '.data/tiny-imagenet-200/train'
    val_dir = '.data/tiny-imagenet-200/val'
    wnids_dir = '.data/tiny-imagenet-200/wnids.txt'
    words_dir = '.data/tiny-imagenet-200/words.txt'

    # 加载类别标签
    with open(wnids_dir, 'r') as f:
        wnids = [line.strip() for line in f]

    with open(words_dir, 'r') as f:
        wnid_to_words = dict(line.split('\t') for line in f)

        # 加载训练集
    train_data = []
    train_labels = []

    for wnid in wnids:
        img_dir = os.path.join(train_dir, wnid, 'images')
        for img_file in os.listdir(img_dir):
            img_path = os.path.join(img_dir, img_file)
            img = Image.open(img_path)
            img = np.array(img)  # 转换为numpy数组
            train_data.append(img)
            train_labels.append(wnid)

            # 加载验证集
    val_data = []
    val_labels = []

    with open(os.path.join(val_dir, 'val_annotations.txt'), 'r') as f:
        for line in f:
            img_file, wnid = line.split('\t')[:2]
            img_path = os.path.join(val_dir, 'images', img_file)
            img = Image.open(img_path)
            img = np.array(img)  # 转换为numpy数组
            val_data.append(img)
            val_labels.append(wnid)

            # 可以将数据进一步转换为模型所需的格式，比如TensorFlow中的tf.data.Dataset或PyTorch中的Dataset类

def random_visualize(imgs, labels, label_names):
    figure = plt.figure(figsize=(len(label_names), 10))
    idxs = list(range(len(imgs)))
    np.random.shuffle(idxs)
    count = [0] * len(label_names)
    for idx in idxs:
        label = labels[idx]
        if count[label] >= 10:
            continue
        if sum(count) > 10 * len(label_names):
            break

        img = imgs[idx]
        label_name = label_names[label]

        subplot_idx = count[label] * len(label_names) + label + 1
        print(label, subplot_idx)
        plt.subplot(10, len(label_names), subplot_idx)
        plt.imshow(img)
        # plt.xticks([])
        # plt.yticks([])
        plt.axis('off')
        if count[label] == 0:
            plt.title(label_name)

        count[label] += 1

    plt.show()


def te():
    # 设定数据集路径
    train_dir = '.data/tiny-imagenet-200/train'
    # val_dir = '.data/tiny-imagenet-200/val'
    wnids_dir = '.data/tiny-imagenet-200/wnids.txt'
    words_dir = '.data/tiny-imagenet-200/words.txt'

    # 加载类别标签
    with open(wnids_dir, 'r') as f:
        wnids = [line.strip() for line in f]

    with open(words_dir, 'r') as f:
        wnid_to_words = dict(line.split('\t') for line in f)
        # print(wnid_to_words)

        # 选择10个类别
    selected_classes = np.random.choice(wnids, 10, replace=False)

    # 创建一个10x10的子图
    fig, axes = plt.subplots(10, 10, figsize=(20, 20))

    for j, wnid in enumerate(selected_classes):
        class_dir = os.path.join(train_dir, wnid, 'images')
        image_files = os.listdir(class_dir)
        selected_images = np.random.choice(image_files, 10, replace=False)
        ww = wnid_to_words[wnid].split(',')

        # print(ww[0], len(wnid_to_words[wnid]), wnid_to_words[wnid])
        axes[0, j].set_title(ww[0])
        for i, img_file in enumerate(selected_images):
            img_path = os.path.join(class_dir, img_file)
            img = Image.open(img_path)
            axes[i, j].imshow(img)
            #
            axes[i, j].axis('off')

    plt.show()

if __name__ == '__main__':
    te()