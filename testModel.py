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
import torchvision.transforms as transforms

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
    parser.add_argument("-f", "--test_file",
                        help="Path to data file.", 
                        metavar="FILE_PATH", 
                        default="landmark_list.txt")
    parser.add_argument("-imgf", "--file_img",
                        help="Path to data file.", 
                        metavar="FILE_PATH", 
                        default="none")                    
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

    model.load_state_dict(torch.load("./models/" + args.model.split()[0]))
    print(evaluate(model, args.test_file, device))
    
    #Attempting to compare output of our neural network to actual values
    UTKFace = CustomImageDataset("testImage.txt", 'UTKFace')
    valid_set = DataLoader(UTKFace, 5, shuffle=True)
    model.eval()

    # Loops through and outputs a prediction based on our model for the location of the nose on people
    # this information is then displayed on a graph
    if args.file_img != 'none':
        with open(args.file_img) as file:
            for line in file:
                imgfile = line.split(".jpg")
                plt.rcParams["figure.figsize"] = [7.00, 3.50]
                plt.rcParams["figure.autolayout"] = True
                im = plt.imread("./UTKFace/" + imgfile[0] + ".jpg.chip.jpg")

                im_tensor = torch.from_numpy(im).float()
                im_tensor = im_tensor.view(1, 3, 200, 200)
                output = model(im_tensor)

                fig, ax = plt.subplots()
                im = ax.imshow(im, extent=[0, 200, 0, 200])
                x = np.array(range(200))
                ax.scatter(output[0,0].item(), output[0,1].item(), ls='dotted', linewidth=2, color='red')
                plt.show()

    

