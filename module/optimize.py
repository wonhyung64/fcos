import tensorflow as tf


def build_optimizer(batch_size, data_num, momentum):
    boundaries = [data_num // batch_size * epoch for epoch in (60, 80)]
    values = [1e-4, 1e-5, 1e-6]
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
    buffer_model,
    weights_decay,
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
    weight_decay_decoupled(model, buffer_model, weights_decay)

    return reg_loss, ctr_loss, clf_loss, total_loss


def weight_decay_decoupled(model, buffer_model, decay_rate):
    # weight decay
    for var, buffer_var in zip(model.trainable_weights, buffer_model.trainable_weights):
        var.assign(var - decay_rate * buffer_var)
    # update buffer model
    for var, buffer_var in zip(model.trainable_weights, buffer_model.trainable_weights):
        buffer_var.assign(var)
