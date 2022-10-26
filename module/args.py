import argparse


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="voc/2007")
    parser.add_argument("--data-dir", type=str, default="/Volumes/LaCie/data")
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--img-size", nargs="+", type=int, default=[512, 512])
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--momentum", type=float, default=0.9)

    try:
        args = parser.parse_args()
    except:
        args = parser.parse_args([])

    return args
