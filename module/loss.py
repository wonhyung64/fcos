import tensorflow as tf


class FCOSBoxLoss(tf.keras.losses.Loss):

    def __init__(self, args):
        super(FCOSBoxLoss, self).__init__(
            reduction="none", name="FCOSLoss"
        )
        self.N = tf.cast(args.batch_size, dtype=tf.float32)
    
    def _compute_iou(self, true, pred):
        pred_l, pred_r, pred_t, pred_b = tf.split(pred, 4, axis=-1)
        gt_l, gt_r, gt_t, gt_b = tf.split(true, 4, axis=-1)

        inter_width = tf.minimum(gt_l, pred_l) + tf.minimum(gt_r, pred_r)
        inter_height = tf.minimum(gt_t, pred_t) + tf.minimum(gt_b, pred_b)
        inter_area = inter_width * inter_height
        union_area = (gt_l + gt_r) * (gt_t + gt_b) + (pred_l + pred_r) * (pred_t + pred_b) - inter_area
        iou = inter_area / union_area

        return iou

    @tf.function
    def call(self, true, pred):
        iou = self._compute_iou(true, pred)

        pos_mask = tf.expand_dims(tf.where(tf.reduce_sum(true, -1) != 0, 1., 0.), -1)
        reg_loss = tf.reduce_sum(-tf.math.log(iou+tf.constant(1e-12)) * pos_mask) / self.N

        return reg_loss


class FCOSCenternessLoss(tf.keras.losses.Loss):

    def __init__(self, args):
        super(FCOSCenternessLoss, self).__init__(
            reduction="none", name="FCOSLoss"
        )
        self.N = tf.cast(args.batch_size, dtype=tf.float32)

    def _bce_loss(self, true, pred):
        return -(true * tf.math.log(pred + tf.constant(1e-12)) + (tf.constant(1.) - true) * tf.math.log(tf.constant(1.) - pred + tf.constant(1e-12)))

    @tf.function
    def call(self, true, pred):
        ctr_loss = tf.reduce_sum(self._bce_loss(true, pred)) / self.N

        return ctr_loss


class FCOSClassificationLoss(tf.keras.losses.Loss):

    def __init__(self, args):
        super(FCOSClassificationLoss, self).__init__(
            reduction="none", name="FCOSLoss"
        )
        self.N = tf.cast(args.batch_size, dtype=tf.float32)
        self.alpha = args.alpha
        self.gamma = args.gamma

    def _bce_loss(self, true, pred):
        return -(true * tf.math.log(pred + tf.constant(1e-12)) + (tf.constant(1.) - true) * tf.math.log(tf.constant(1.) - pred + tf.constant(1e-12)))

    @tf.function
    def call(self, true, pred):
        clf_loss = self._bce_loss(true, pred)
        alpha = tf.where(tf.equal(true, 1.0), self.alpha, (1.0 - self.alpha))
        pt = tf.where(tf.equal(true, 1.0), pred, tf.constant(1.) - pred)
        clf_loss = tf.reduce_sum(alpha * tf.pow(tf.constant(1.) - pt, self.gamma) * clf_loss) / self.N

        return clf_loss
