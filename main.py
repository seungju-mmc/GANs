import os
import torch
import argparse

import numpy as np

from dataset import trainloader, testloader, imShow
from baseline.baseline.utils import jsonParser
from network import WGANnn


parser = argparse.ArgumentParser(
    description="WGAN"
)

parser.add_argument(
    '--path',
    '-p',
    type=str,
    default="./cfg/WGAN.json",
    help="file path for configuration."
)

parser.add_argument(
    '--train',
    '-t',
    action='store_true',
    default=True,
    help="specify the train mode."
)
args = parser.parse_args()


if __name__ == "__main__":
    path = args.path
    if os.path.isfile(path):
        print("Loading Configuration File.")
    else:
        RuntimeError("the path is not valid.")
