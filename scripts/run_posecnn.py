import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--train", action="store_true")
parser.add_argument("--eval", action="store_true")
parser.add_argument("--data-dir", default="data", help="PROPS-Pose dataset root")
parser.add_argument("--device", default=None, help="Torch device; defaults to CUDA when available")
parser.add_argument("--batch-size", type=int, default=2)
parser.add_argument("--num-classes", type=int, default=10)
args = parser.parse_args()

if __name__ == "__main__":
    from pose_cnn import train_posecnn, eval_posecnn
    from utils.posecnn_utils import reset_seed

    reset_seed(0)
    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.train:
        print("Training PoseCNN")
        train_posecnn(args.data_dir, batch_size=args.batch_size, num_classes=args.num_classes, device=device)
    else:
        print("Evaluating PoseCNN")
        eval_posecnn(args.data_dir, batch_size=args.batch_size, num_classes=args.num_classes, device=device)
