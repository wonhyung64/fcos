#%%
from module.dataset import load_dataset
import tensorflow as tf
import numpy as np
import sys
import os
import math
import tensorflow_addons as tfa

# %%
def preprocess(dataset, img_size, num_classes, split=None):
    img = dataset["image"]
    img = tf.image.resize(img, img_size, "bicubic")
    img = tf.image.per_image_standardization(img)
    gt_boxes = dataset["objects"]["bbox"]
    gt_labels = dataset["objects"]["label"]
    gt_labels = tf.cast(tf.expand_dims(gt_labels, axis=-1), dtype=tf.float32)
    ground_truth = tf.concat([gt_boxes, gt_labels], axis=-1) 

    if split == "train":
        gt_regs, gt_ctrs, gt_clfs = label_encoder(ground_truth, img_size, num_classes)
        return img, gt_regs, gt_ctrs, gt_clfs
    else:
        return img, ground_truth

def label_encoder(gt, img_size, num_classes):
    s3, s4, s5, s6, s7 = 8, 16, 32, 64, 128
    gt_h = gt[..., 2] - gt[..., 0] 
    gt_w = gt[..., 3] - gt[...,1]
    gt_size = tf.sqrt(gt_h * gt_w * math.prod(img_size))

    g3 = tf.boolean_mask(gt, gt_size <= 64.)
    g4 = tf.boolean_mask(gt, tf.cast(gt_size >= 64., tf.float32)*tf.cast(gt_size <= 128., tf.float32) > 0.)
    g5 = tf.boolean_mask(gt, tf.cast(gt_size >= 128., tf.float32)*tf.cast(gt_size <= 256., tf.float32) > 0.)
    g6 = tf.boolean_mask(gt, tf.cast(gt_size >= 256., tf.float32)*tf.cast(gt_size <= 512., tf.float32) > 0.)
    g7 = tf.boolean_mask(gt, gt_size >= 512.)

    gt_regs, gt_ctrs, gt_clfs = [], [], []
    for g_, s_ in [(g3, s3), (g4, s4), (g5, s5), (g6, s6), (g7, s7)]:
        class_id = tf.cast(g_[..., 4], dtype=tf.int32)
        feature_map_shape = tf.cast(img_size, dtype=tf.float32) / s_
        if len(class_id) == 0:
            gt_reg = tf.zeros(tf.concat([tf.cast(feature_map_shape,dtype=tf.int32), [4]], axis=-1), dtype=tf.float32)
            gt_ctr = tf.zeros(tf.cast(feature_map_shape, dtype=tf.int32), dtype=tf.float32)
            gt_clf = tf.zeros(tf.concat([tf.cast(feature_map_shape, dtype=tf.int32), [num_classes]], axis=-1), dtype=tf.float32)

        else:
            gbbox_y1, gbbox_x1, gbbox_y2, gbbox_x2 = tf.split(
                g_[...,:-1] * tf.tile(feature_map_shape, [2]), 4, axis=-1
                )

            gbbox_y1 = tf.reshape(gbbox_y1, [1, 1, -1])
            gbbox_x1 = tf.reshape(gbbox_x1, [1, 1, -1])
            gbbox_y2 = tf.reshape(gbbox_y2, [1, 1, -1])
            gbbox_x2 = tf.reshape(gbbox_x2, [1, 1, -1])

            num_g = tf.shape(gbbox_y1)[-1]

            h_ = tf.range(0., feature_map_shape[0])
            w_ = tf.range(0., feature_map_shape[1])
            grid_y_, grid_x_ = tf.meshgrid(h_, w_)

            grid_y = tf.expand_dims(grid_y_, -1)
            grid_x = tf.expand_dims(grid_x_, -1)
            grid_y = tf.tile(grid_y, [1, 1, num_g])
            grid_x = tf.tile(grid_x, [1, 1, num_g])

            dist_l = grid_x - gbbox_x1
            dist_r = gbbox_x2 - grid_x
            dist_t = grid_y - gbbox_y1
            dist_b = gbbox_y2 - grid_y

            grid_y_mask = tf.cast(dist_t > 0., tf.float32) * tf.cast(dist_b > 0., tf.float32)
            grid_x_mask = tf.cast(dist_l > 0., tf.float32) * tf.cast(dist_r > 0., tf.float32)
            heatmask = grid_y_mask * grid_x_mask

            dist_l *= heatmask
            dist_r *= heatmask
            dist_t *= heatmask
            dist_b *= heatmask
            loc = tf.reduce_max(heatmask, axis=-1)

            dist_area = (dist_l + dist_r) * (dist_t + dist_b)
            dist_area_ = dist_area + (1. - heatmask) * 1e8
            dist_area_min = tf.reduce_min(dist_area_, axis=-1, keepdims=True)
            dist_mask = tf.cast(tf.equal(dist_area, dist_area_min), tf.float32) * tf.expand_dims(loc, axis=-1)

            dist_l *= dist_mask
            dist_r *= dist_mask
            dist_t *= dist_mask
            dist_b *= dist_mask
            dist_l = tf.reduce_max(dist_l, axis=-1)
            dist_r = tf.reduce_max(dist_r, axis=-1)
            dist_t = tf.reduce_max(dist_t, axis=-1)
            dist_b = tf.reduce_max(dist_b, axis=-1)
            gt_reg = tf.stack([dist_l, dist_r, dist_t, dist_b], axis=-1)

            lr_min = tf.minimum(dist_l, dist_r)
            tb_min = tf.minimum(dist_t, dist_b)
            lr_max = tf.maximum(dist_l, dist_r)
            tb_max = tf.maximum(dist_t, dist_b)
            gt_ctr = tf.sqrt(lr_min*tb_min/(lr_max*tb_max+1e-12))

            zero_like_heat = tf.expand_dims(tf.zeros(tf.cast(feature_map_shape, dtype=tf.int32), dtype=tf.float32), axis=-1)
            heatmap_gt = []
            for i in range(num_classes):
                exist_i = tf.equal(class_id, i)
                heatmask_i = tf.boolean_mask(heatmask, exist_i, axis=2)
                heatmap_i = tf.cond(
                    tf.equal(tf.shape(heatmask_i)[-1], 0),
                    lambda: zero_like_heat,
                    lambda: tf.reduce_max(heatmask_i, axis=2, keepdims=True)
                )
                heatmap_gt.append(heatmap_i)
            gt_clf = tf.concat(heatmap_gt, axis=-1)

        gt_regs.append(gt_reg)
        gt_ctrs.append(gt_ctr)
        gt_clfs.append(gt_clf)

    gt_regs = tf.concat([tf.reshape(gt_reg, [-1, 4]) for gt_reg in gt_regs], axis=0)
    gt_ctrs = tf.concat([tf.reshape(gt_ctr, [-1, 1]) for gt_ctr in gt_ctrs], axis=0)
    gt_clfs = tf.concat([tf.reshape(gt_clf, [-1, num_classes]) for gt_clf in gt_clfs], axis=0)
    
    return gt_regs, gt_ctrs, gt_clfs


def build_dataset(datasets, batch_size, img_size, num_classes):
    train_set, valid_set, test_set = datasets

    train_set = train_set.map(lambda x: preprocess(x, img_size, num_classes, split="train"))
    valid_set = valid_set.map(lambda x: preprocess(x, img_size, num_classes))
    test_set = test_set.map(lambda x: preprocess(x, img_size, num_classes))

    train_set = train_set.batch(batch_size).repeat()
    valid_set = valid_set.batch(1).repeat()
    test_set = test_set.batch(1).repeat()

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

#%%
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, ReLU, Lambda, Layer
class FCOS(Model):
    def __init__(self, num_classes, backbone=None, **kwargs):
        super(FCOS, self).__init__(name="FCOS", **kwargs)
        self.num_classes = num_classes
        self.fpn = FeaturePyramid(backbone)

        prior_probability = tf.constant_initializer(-np.log((1 - 0.01) / 0.01))
        self.cls_head = build_head(self.num_classes + 1, prior_probability)
        self.box_head = build_head(4, "zeros")
        self.divider = Lambda(lambda x: [x[..., :-1], x[..., -1:]])


    @tf.function
    def call(self, image, training=False):
        features = self.fpn(image, training=training)
        N = tf.shape(image)[0]
        cls_outputs = []
        ctr_outputs = []
        box_outputs = []
        for feature in features:
            box_outputs.append(tf.reshape(tf.exp(self.box_head(feature)), [N, -1, 4]))
            cls_output = self.cls_head(feature)
            cls_output, ctr_output = self.divider(cls_output)
            cls_outputs.append(
                tf.reshape(tf.sigmoid(cls_output), [N, -1, self.num_classes])
            )
            ctr_outputs.append(
                tf.reshape(tf.sigmoid(ctr_output), [N, -1, 1])
            )
        box_outputs = tf.concat(box_outputs, axis=1)
        ctr_outputs = tf.concat(ctr_outputs, axis=1)
        cls_outputs = tf.concat(cls_outputs, axis=1)

        return (box_outputs, ctr_outputs, cls_outputs)


def get_backbone():
    backbone = tf.keras.applications.ResNet50(
        include_top=False, input_shape=[None, None, 3]
    )
    c3_output, c4_output, c5_output = [
        backbone.get_layer(layer_name).output
        for layer_name in ["conv3_block4_out", "conv4_block6_out", "conv5_block3_out"]
    ]

    return tf.keras.Model(
        inputs=[backbone.inputs], outputs=[c3_output, c4_output, c5_output]
    )


def build_head(output_filters, bias_init):
    head = tf.keras.Sequential([tf.keras.Input(shape=[None, None, 256])])
    kernel_init = tf.initializers.RandomNormal(0.0, 0.01)
    for _ in range(4):
        head.add(
            Conv2D(256, 3, padding="same", kernel_initializer=kernel_init)
        )
        head.add(ReLU())
    head.add(
        Conv2D(
            output_filters,
            3,
            1,
            padding="same",
            kernel_initializer=kernel_init,
            bias_initializer=bias_init,
        )
    )

    return head

class FeaturePyramid(Layer):

    def __init__(self, backbone=None, **kwargs):
        super(FeaturePyramid, self).__init__(name="FeaturePyramid", **kwargs)
        self.backbone = backbone if backbone else get_backbone()
        self.conv_c3_1x1 = Conv2D(256, 1, 1, "same")
        self.conv_c4_1x1 = Conv2D(256, 1, 1, "same")
        self.conv_c5_1x1 = Conv2D(256, 1, 1, "same")
        self.conv_c3_3x3 = Conv2D(256, 3, 1, "same")
        self.conv_c4_3x3 = Conv2D(256, 3, 1, "same")
        self.conv_c5_3x3 = Conv2D(256, 3, 1, "same")
        self.conv_c6_3x3 = Conv2D(256, 3, 2, "same")
        self.conv_c7_3x3 = Conv2D(256, 3, 2, "same")
        self.upsample_2x = CustomUpSampling2D(2)

    @tf.function
    def call(self, images, training=False):
        c3_output, c4_output, c5_output = self.backbone(images, training=training)
        p3_output = self.conv_c3_1x1(c3_output)
        p4_output = self.conv_c4_1x1(c4_output)
        p5_output = self.conv_c5_1x1(c5_output)
        p4_output = p4_output + self.upsample_2x(p5_output)
        p3_output = p3_output + self.upsample_2x(p4_output)
        p3_output = self.conv_c3_3x3(p3_output)
        p4_output = self.conv_c4_3x3(p4_output)
        p5_output = self.conv_c5_3x3(p5_output)
        p6_output = self.conv_c6_3x3(c5_output)
        p7_output = self.conv_c7_3x3(tf.nn.relu(p6_output))

        return p3_output, p4_output, p5_output, p6_output, p7_output

class CustomUpSampling2D(tf.keras.layers.Layer):
    def __init__(self, size):
        super(CustomUpSampling2D, self).__init__()
        if type(size) is not tuple and type(size) is not list:
            size = (size, size)
        self.size = size

    def build(self, input_shape):
        pass

    def call(self, input):
        return tf.repeat(tf.repeat(input, self.size[0], axis=1), self.size[1], axis=2)


class DecodePredictions(tf.keras.layers.Layer):

    def __init__(
        self,
        img_size,
        num_classes=20,
        confidence_threshold=0.05,
        nms_iou_threshold=0.5,
        max_detections_per_class=100,
        max_detections=100,
        **kwargs
        ):
        super(DecodePredictions, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.max_detections_per_class = max_detections_per_class
        self.max_detections = max_detections
        self.strides = [8, 16, 32, 64, 128]
        self.grid_y, self.grid_x, self.grid_s = build_grids(img_size, self.strides)

    @tf.function
    def call(self, pred_regs, pred_ctrs, pred_clfs):
        pred_score = pred_clfs * pred_ctrs
        
        pred_x1 = self.grid_x - pred_regs[...,0:1]
        pred_x2 = self.grid_x + pred_regs[...,1:2]
        pred_y1 = self.grid_y - pred_regs[...,2:3]
        pred_y2 = self.grid_y + pred_regs[...,3:4]

        pred_bbox = tf.concat([pred_y1, pred_x1, pred_y2, pred_x2], axis=-1) * self.grid_s

        final_bboxes, final_labels, final_scores, _ =  tf.image.combined_non_max_suppression(
            tf.expand_dims(pred_bbox, axis=2),
            pred_score,
            self.max_detections_per_class,
            self.max_detections,
            self.nms_iou_threshold,
            self.confidence_threshold,
            clip_boxes=False,
        )
        idx = final_scores != 0
        final_bboxes = final_bboxes[idx]
        final_labels = final_labels[idx]
        final_scores = final_scores[idx]

        return final_bboxes, final_labels, final_scores


def build_grids(img_size, strides):
    grid_y, grid_x, grid_s = [], [], []
    for s_ in strides:
        feature_map_shape = tf.cast(img_size, dtype=tf.float32) / s_
        h_ = tf.range(0., feature_map_shape[0])
        w_ = tf.range(0., feature_map_shape[1])
        grid_y_, grid_x_ = tf.meshgrid(h_, w_)
        grid_s_ = tf.ones_like(grid_y_, dtype=tf.float32) * s_
        grid_y.append(tf.reshape(tf.expand_dims(grid_y_, axis=-1), [-1, 1]))
        grid_x.append(tf.reshape(tf.expand_dims(grid_x_, axis=-1), [-1, 1]))
        grid_s.append(tf.reshape(tf.expand_dims(grid_s_, axis=-1), [-1, 1]))
    grid_y = tf.concat(grid_y, axis=0)
    grid_x = tf.concat(grid_x, axis=0)
    grid_s = tf.concat(grid_s, axis=0)

    return grid_y, grid_x, grid_s


def compute_loss(gt_regs, gt_ctrs, gt_clfs, pred_regs, pred_ctrs, pred_clfs, alpha, gamma):
    N = tf.cast(tf.shape(pred_regs)[0], dtype=tf.float32)

    pred_l, pred_r, pred_t, pred_b = tf.split(pred_regs, 4, axis=-1)
    gt_l, gt_r, gt_t, gt_b = tf.split(gt_regs, 4, axis=-1)

    inter_width = tf.minimum(gt_l, pred_l) + tf.minimum(gt_r, pred_r)
    inter_height = tf.minimum(gt_t, pred_t) + tf.minimum(gt_b, pred_b)
    inter_area = inter_width * inter_height
    union_area = (gt_l + gt_r) * (gt_t + gt_b) + (pred_l + pred_r) * (pred_t + pred_b) - inter_area
    iou = inter_area / union_area
    pos_cond = tf.not_equal(gt_ctrs, tf.constant(0.0))
    pos_mask = tf.cast(pos_cond, dtype=tf.float32)
    reg_loss = tf.reduce_sum(-tf.math.log(iou+1e-12) * pos_mask) / N

    ctr_loss = -tf.reduce_sum(gt_ctrs * tf.math.log(pred_ctrs + 1e-12) + (1 - gt_ctrs) * tf.math.log(1 - pred_ctrs + 1e-12)) / N

    bce = -(gt_clfs * tf.math.log(pred_clfs + 1e-12) + (1 - gt_clfs) * tf.math.log(1 - pred_clfs + 1e-12))
    alpha = tf.where(tf.equal(gt_clfs, 1.0), alpha, (1.0 - alpha))
    pt = tf.where(tf.equal(gt_clfs, 1.0), pred_clfs, 1 - pred_clfs)
    clf_loss = tf.reduce_sum(alpha * tf.pow(1.0 - pt, gamma) * bce) / N

    return reg_loss, ctr_loss, clf_loss


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="voc/2007")
    parser.add_argument("--data-dir", type=str, default="/Volumes/LaCie/data")
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--img-size", nargs="+", type=int, default=[512, 512])
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--gamma", type=float, default=2.0)

    try:
        args = parser.parse_args()
    except:
        args = parser.parse_args([])

    return args

# %%
import argparse


args = build_args()
datasets, labels, train_num, valid_num, test_num = load_dataset(name=args.name, data_dir=args.data_dir)
train_set, valid_set, test_set = build_dataset(datasets, args.batch_size, args.img_size, labels.num_classes)

fcos = FCOS(labels.num_classes)
fcos.build([None, 512,512,3])
fcos.summary()

decoder = DecodePredictions(args.img_size)

#%%
alpha = args.alpha
gamma = args.gamma

img, gt_regs, gt_ctrs, gt_clfs = next(train_set)
pred_regs, pred_ctrs, pred_clfs = fcos(img)
reg_loss, ctr_loss, clf_loss = compute_loss(gt_regs, gt_ctrs, gt_clfs, pred_regs, pred_ctrs, pred_clfs, alpha, gamma)
final_bboxes, final_labels, final_scores = decoder(pred_regs, pred_ctrs, pred_clfs)
