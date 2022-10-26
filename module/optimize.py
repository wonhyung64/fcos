import tensorflow as tf


def build_optimizer(batch_size, data_num, momentum):
    boundaries = [data_num // batch_size * epoch for epoch in (1, 50, 60, 70)]
    values = [1e-5, 1e-3, 1e-4, 1e-6, 1e-7]
    learning_rate_fn = tf.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=boundaries, values=values
    )

    optimizer = tf.optimizers.SGD(learning_rate=learning_rate_fn, momentum=momentum)

    return optimizer


@tf.function
def forward_backward(
    img,
    true,
    model,
    reg_fn,
    ctr_fn,
    clf_fn,
    optimizer,
):
    gt_regs, gt_ctrs, gt_clfs = true
    with tf.GradientTape(persistent=True) as tape:
        pred_regs, pred_ctrs, pred_clfs = model(img)
        reg_loss = reg_fn(gt_regs, pred_regs)
        ctr_loss = ctr_fn(gt_ctrs, pred_ctrs)
        clf_loss = clf_fn(gt_clfs, pred_clfs)
        total_loss = tf.reduce_sum([reg_loss, ctr_loss, clf_loss])

    grads = tape.gradient(total_loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    return reg_loss, ctr_loss, clf_loss, total_loss