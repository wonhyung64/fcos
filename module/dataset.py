import os
import tensorflow as tf
import tensorflow_datasets as tfds
from .target import label_encoder


def load_dataset(name, data_dir):
    train1, dataset_info = tfds.load(
        name=name, split="train", data_dir=f"{data_dir}/tfds", with_info=True
    )
    train2 = tfds.load(
        name=name,
        split="validation[100:]",
        data_dir=f"{data_dir}/tfds",
    )
    valid_set = tfds.load(
        name=name,
        split="validation[:100]",
        data_dir=f"{data_dir}/tfds",
    )
    test_set = tfds.load(
        name=name,
        split="train[:10%]",
        data_dir=f"{data_dir}/tfds",
    )
    train_set = train1.concatenate(train2)

    train_num, valid_num, test_num = load_data_num(
        name, data_dir, train_set, valid_set, test_set
    )

    try:
        labels = dataset_info.features["labels"]
    except:
        labels = dataset_info.features["objects"]["label"]

    return (train_set, valid_set, test_set), labels, train_num, valid_num, test_num


def load_data_num(name, data_dir, train_set, valid_set, test_set):
    data_nums = []
    for dataset, dataset_name in (
        (train_set, "train"),
        (valid_set, "validation"),
        (test_set, "test"),
    ):
        data_num_dir = f"{data_dir}/data_chkr/{''.join(char for char in name if char.isalnum())}_{dataset_name}_num.txt"

        if not (os.path.exists(data_num_dir)):
            data_num = build_data_num(dataset, dataset_name)
            with open(data_num_dir, "w") as f:
                f.write(str(data_num))
                f.close()
        else:
            with open(data_num_dir, "r") as f:
                data_num = int(f.readline())
        data_nums.append(data_num)

    return data_nums


def build_data_num(dataset, dataset_name):
    num_chkr = iter(dataset)
    data_num = 0
    print(f"\nCounting number of {dataset_name} data\n")
    while True:
        try:
            next(num_chkr)
        except:
            break
        data_num += 1

    return data_num


def build_dataset(datasets, batch_size, img_size, num_classes):
    train_set, valid_set, test_set = datasets

    train_set = train_set.map(lambda x: preprocess(x, img_size, num_classes, split="train"))
    valid_set = valid_set.map(lambda x: preprocess(x, img_size, num_classes))
    test_set = test_set.map(lambda x: preprocess(x, img_size, num_classes))

    train_set = train_set.batch(batch_size).repeat()
    valid_set = valid_set.repeat()
    test_set = test_set.repeat()

    autotune = tf.data.AUTOTUNE
    train_set = train_set.apply(tf.data.experimental.ignore_errors())
    train_set = train_set.prefetch(autotune)
    valid_set = valid_set.apply(tf.data.experimental.ignore_errors())
    valid_set = valid_set.prefetch(autotune)
    test_set = test_set.apply(tf.data.experimental.ignore_errors())
    test_set = test_set.prefetch(autotune)

    train_set = iter(train_set)
    valid_set = iter(valid_set)
    test_set = iter(test_set)

    return train_set, valid_set, test_set


def preprocess(dataset, img_size, num_classes, split=None):
    img = dataset["image"]
    img = tf.image.resize(img, img_size, "bicubic")
    # img = tf.image.per_image_standardization(img)
    gt_boxes = dataset["objects"]["bbox"]
    gt_labels = dataset["objects"]["label"]
    gt_labels = tf.cast(tf.expand_dims(gt_labels, axis=-1), dtype=tf.float32)
    ground_truth = tf.concat([gt_boxes, gt_labels], axis=-1) 

    if split == "train":
        gt_regs, gt_ctrs, gt_clfs = label_encoder(ground_truth, img_size, num_classes)
        return img, gt_regs, gt_ctrs, gt_clfs
    else:
        return img, ground_truth
