import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from train import evaluate
from net import convNN, convNN2

import sys
import time
import curses

import numpy as np

from torch.utils.data import DataLoader
from argparse import ArgumentParser

import matplotlib.pyplot as plt

from dataset import CustomImageDataset

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-m", "--model",
                        help="Choose which model structure to use.",
                        default="convNN2",
                        metavar="MODEL_NAME")
    parser.add_argument("-f", "--train_file",
                        help="Path to data file.", 
                        metavar="FILE_PATH", 
                        default="landmark_list.txt")
    parser.add_argument("--cuda",
                        help="Add this argument to run the code using GPU acceleration.",
                        action="store_true")
    
    args = parser.parse_args()

    device = "cpu"
    if args.cuda and torch.cuda.is_available():
        device = "cuda"
    if substr()
    model = convNN2()
    model.load_state_dict(torch.load("./models/" + args.model))
    print(evaluate(model, args.train_file, device))


