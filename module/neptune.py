import sys
import subprocess
try:
    import neptune.new as neptune
except:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "neptune-client"])
    import neptune.new as neptune


def plugin_neptune(NEPTUNE_API_KEY, NEPTUNE_PROJECT, args):
    run = neptune.init(
        project=NEPTUNE_PROJECT,
        api_token=NEPTUNE_API_KEY,
        mode="async",
    )
    run["sys/name"] = "fcos-optimization"
    run["sys/tags"].add([f"{key}: {value}" for key, value in args._get_kwargs()])

    return run


def record_train_loss(run, reg_loss, ctr_loss, clf_loss, total_loss):
    run["train/loss/reg_loss"].log(reg_loss.numpy())
    run["train/loss/ctr_loss"].log(ctr_loss.numpy())
    run["train/loss/clf_loss"].log(clf_loss.numpy())
    run["train/loss/total_loss"].log(total_loss.numpy())


def record_result(run, weights_dir, train_time, mean_ap, mean_test_time):
    res = {
        "mean_ap": "%.3f" % (mean_ap.numpy()),
        "train_time": train_time,
        "inference_time": "%.2fms" % (mean_test_time.numpy()),
    }
    run["results"] = res
    run["model"].upload(f"{weights_dir}.h5")
