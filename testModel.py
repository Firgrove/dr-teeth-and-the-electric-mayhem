import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from train import evaluate
from net import convNN, convNN2
from dataset import CustomImageDataset

import sys
import time
import curses

import numpy as np

from torch.utils.data import DataLoader
from argparse import ArgumentParser

import matplotlib.pyplot as plt

from torchvision.io import read_image

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

    if "convNN_" in args.model:
        model = convNN()
    else:
        model = convNN2()

    model.load_state_dict(torch.load("./models/" + args.model))
    print(evaluate(model, args.train_file, device))
    
    #Attempting to compare output of our neural network to actual values
    UTKFace = CustomImageDataset("testImage.txt", 'UTKFace')
    valid_set = DataLoader(UTKFace, 500)
    model.eval()

    with torch.no_grad():
        for images, _, _, _, landmarks in valid_set:
            images, landmarks = images.to(device), landmarks.to(device)
            outputs = model(images)


    print(outputs)
    print(outputs[0][0].item())

    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True
    im = plt.imread("./UTKFace/2_1_2_20161219140650888.jpg.chip.jpg")
    fig, ax = plt.subplots()
    im = ax.imshow(im, extent=[0, 100, 0, 100])
    x = np.array(range(100))
    ax.scatter(outputs[0,0].item()*200 , outputs[0,1].item()*200, ls='dotted', linewidth=2, color='red')
    plt.show()


