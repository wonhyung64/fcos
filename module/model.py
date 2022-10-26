import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, ReLU, Lambda, Layer


def build_model(args, labels):
    input_shape = [None] + args.img_size + [3]
    fcos = FCOS(labels.num_classes)
    fcos.build(input_shape)

    buffer_model = FCOS(labels.num_classes)
    buffer_model.build(input_shape)
    buffer_model.set_weigths(fcos.get_weights())

    return fcos, buffer_model


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
            cls_outputs.append(
                tf.reshape(tf.sigmoid(cls_output), [N, -1, self.num_classes+1])
            )
        box_outputs = tf.concat(box_outputs, axis=1)
        cls_outputs = tf.concat(cls_outputs, axis=1)
        clf_outputs, ctr_outputs = self.divider(cls_outputs)

        return (box_outputs, ctr_outputs, clf_outputs)


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
        head.add(tfa.layers.GroupNormalization(32))
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
        self.grid_y, self.grid_x, self.grid_s = self._build_grids(img_size, self.strides)

    def _build_grids(self, img_size, strides):
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
