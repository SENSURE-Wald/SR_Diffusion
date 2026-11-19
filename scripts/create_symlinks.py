import argparse
import os
import pathlib


def read_and_symlink(split_file, split, input_path):
    base_path_s2 = input_path / f"{split}/s2"
    base_path_dop = input_path / f"{split}/dop"

    os.makedirs(base_path_s2, exist_ok=True)
    os.makedirs(base_path_dop, exist_ok=True)
    with open(split_file, "r") as f:
        for line in f.readlines():
            src_dop = pathlib.Path(line.strip())
            src_s2 = input_path / f"s2/{src_dop.name}"
            s2_path = base_path_s2 / src_dop.name
            dop_path = base_path_dop / src_dop.name

            os.symlink(src=src_s2, dst=s2_path)
            os.symlink(src=src_dop, dst=dop_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("INPUT", type=pathlib.Path, default=pathlib.Path("/big_storage/data/dop/HE"))

    args = parser.parse_args()

    train_split = args.INPUT / "train_split.txt"
    val_split = args.INPUT / "val_split.txt"
    test_split = args.INPUT / "test_split.txt"

    os.makedirs(args.INPUT / "train", exist_ok=True)
    os.makedirs(args.INPUT / "val", exist_ok=True)
    os.makedirs(args.INPUT / "test", exist_ok=True)
    read_and_symlink(train_split, "train", args.INPUT)
    read_and_symlink(val_split, "val", args.INPUT)
    read_and_symlink(test_split, "test", args.INPUT)
