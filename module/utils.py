import os
import time
import tensorflow as tf
import neptune.new as neptune
from tqdm import tqdm
from .args import build_args
from .neptune import plugin_neptune, record_train_loss
from .ap import calculate_ap_const
from .draw import draw_output
from .optimize import forward_backward


def initialize_process(NEPTUNE_API_KEY, NEPTUNE_PROJECT):
    args = build_args()
    os.makedirs(f"{args.data_dir}/data_chkr", exist_ok=True)
    run = plugin_neptune(NEPTUNE_API_KEY, NEPTUNE_PROJECT, args)

    experiment_name = run.get_run_url().split("/")[-1].replace("-", "_")
    model_name = NEPTUNE_PROJECT.split("-")[-1]
    experiment_dir = f"./model_weights/{model_name}"
    os.makedirs(experiment_dir, exist_ok=True)
    weights_dir = f"{experiment_dir}/{experiment_name}"

    return args, run, weights_dir


def train(
    run,
    epochs,
    batch_size,
    train_num,
    valid_num,
    train_set,
    valid_set,
    labels,
    model,
    buffer_model,
    weights_decay,
    decoder,
    reg_fn,
    ctr_fn,
    clf_fn,
    optimizer,
    weights_dir,
):
    best_mean_ap = 0
    start_time = time.time()
    for epoch in range(epochs):
        epoch_progress = tqdm(range(train_num // batch_size))
        for _ in epoch_progress:
            img, gt_regs, gt_ctrs, gt_clfs = next(train_set)
            reg_loss, ctr_loss, clf_loss, total_loss = forward_backward(
                img, (gt_regs, gt_ctrs, gt_clfs), model, buffer_model, weights_decay, reg_fn, ctr_fn, clf_fn, optimizer
                )
            record_train_loss(run, reg_loss, ctr_loss, clf_loss, total_loss)

            epoch_progress.set_description(
                "Epoch {}/{} | reg {:.4f}, ctr {:.4f}, cls {:.4f}, total {:.4f}".format(
                    epoch + 1,
                    epochs,
                    reg_loss.numpy(),
                    ctr_loss.numpy(),
                    clf_loss.numpy(),
                    total_loss.numpy(),
                )
            )
        mean_ap, _ = evaluate(run, valid_set, valid_num, model, decoder, labels, "validation")
        run["validation/mAP"].log(mean_ap.numpy())

        if mean_ap.numpy() > best_mean_ap:
            best_mean_ap = mean_ap.numpy()
            model.save_weights(f"{weights_dir}.h5")

    train_time = time.time() - start_time

    return train_time


def evaluate(run, dataset, dataset_num, model, decoder, labels, split, colors=None):
    eval_times = []
    aps = []
    eval_progress = tqdm(range(dataset_num))
    for step in eval_progress:
        start_time = time.time()

        img, gt = next(dataset)
        gt_boxes, gt_labels = gt[..., :-1], gt[..., -1]
        pred_reg, pred_ctr, pred_clf = model(tf.expand_dims(img, 0))
        final_bboxes, final_labels, final_scores = decoder(pred_reg, pred_ctr, pred_clf)

        eval_time = time.time() - start_time
        eval_times.append(eval_time)

        ap = calculate_ap_const(
            final_bboxes,
            final_labels,
            gt_boxes, gt_labels,
            labels.num_classes, mAP_threshold=0.5
            )
        eval_progress.set_description(f"{split} | Average_Precision {round(ap.numpy(),4)}")
        aps.append(ap)

        if split == "test" and step <= 20:
            run["outputs"].log(
                neptune.types.File.as_image(
                    draw_output(img, final_bboxes, final_labels, final_scores, labels, colors)
                )
            )

    mean_ap = tf.reduce_mean(aps)
    mean_evaltime = tf.reduce_mean(eval_times)

    return mean_ap, mean_evaltime
