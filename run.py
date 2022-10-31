#%%
import tensorflow as tf
from module.dataset import load_dataset, build_dataset
from module.model import build_model, DecodePredictions
from module.loss import FCOSBoxLoss, FCOSCenternessLoss, FCOSClassificationLoss
from module.optimize import build_optimizer, forward_backward
from module.utils import train, evaluate, initialize_process
from module.neptune import record_result
from module.variable import NEPTUNE_API_KEY, NEPTUNE_PROJECT

from module.ap import calculate_ap_const
from module.draw import draw_output
from module.args import build_args

args, run, weights_dir = initialize_process(NEPTUNE_API_KEY, NEPTUNE_PROJECT)

datasets, labels, train_num, valid_num, test_num = load_dataset(name=args.name, data_dir=args.data_dir)
train_set, valid_set, test_set = build_dataset(datasets, args.batch_size, args.img_size, labels.num_classes)
colors = tf.random.uniform((labels.num_classes, 4), maxval=256, dtype=tf.int32)

model, buffer_model = build_model(args, labels)
decoder = DecodePredictions(args.img_size)
reg_fn = FCOSBoxLoss(args)
ctr_fn = FCOSCenternessLoss(args)
clf_fn = FCOSClassificationLoss(args)
optimizer = build_optimizer(args.batch_size, train_num, args.momentum)

#%%
train_time = train(run, args.epochs, args.batch_size,
    train_num, valid_num, train_set, valid_set, labels,
    model, buffer_model, args.weights_decay, decoder,
    reg_fn, ctr_fn, clf_fn, optimizer, weights_dir)

model.load_weights(f"{weights_dir}.h5")
mean_ap, mean_evaltime = evaluate(run, test_set, test_num, model, decoder, labels, "test", colors)
record_result(run, weights_dir, train_time, mean_ap, mean_evaltime)

'''
#%%
model.load_weights("/Users/wonhyung64/Downloads/MOD3_36.h5")
#%%
img, gt_regs, gt_ctrs, gt_clfs = next(train_set)
tf.where(gt_regs != 0)
gt_regs[0, 4966]

pred_regs, pred_ctrs, pred_clfs = model(img[0:1])
final_bboxes, final_scores, final_labels = decoder(pred_regs, pred_ctrs, pred_clfs)
draw_output(img[0], final_bboxes, final_labels, final_scores, labels, colors)
tf.keras.utils.array_to_img(img[15])
labels.names
'''
