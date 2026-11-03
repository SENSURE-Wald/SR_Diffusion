import argparse
from pathlib import Path

parser = argparse.ArgumentParser()

parser.add_argument("INPUT", type=Path)
parser.add_argument("OUTPUT", type=Path)
parser.add_argument("--train", help="Fraction of training data in [0, 1]", default=0.8, type=float)
parser.add_argument("--test", help="Fraction of testing data in [0, 1]", default=0.1, type=float)
parser.add_argument("--val", help="Fraction of validationb data in [0, 1]", default=0.1, type=float)


args = parser.parse_args()

if args.train + args.test + args.val != 1:
    raise ValueError("train + test + val fractions should be equal to 1.")
