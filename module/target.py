import math
import tensorflow as tf


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
