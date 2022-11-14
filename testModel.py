import curses
import json
import sys
import time
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from dataset import CustomImageDataset
from net import convNN, convNN2
from torch.utils.data import DataLoader
from torchvision.io import read_image
from train import evaluate

def get_coords(landmarks: list) -> torch.Tensor:
    landmarks = landmarks[1:7]
    landmarks = [float(i) for i in landmarks]
    landmarks = torch.tensor(landmarks)
    return landmarks.reshape(3,2)

def generate_images(train_dataloader, axs_flat):
    with torch.no_grad():
            for i, (image, _, _, _, labels) in enumerate(train_dataloader):
                output = model(image)
                output = output.reshape(3,2)
                image = image.squeeze()
                image = image.permute(1, 2, 0)    #Default was 3,200,200
                im = axs_flat[i].imshow(image)
                x = np.array(range(200))

                land_idx = [8, 30, 39]
                labels = labels.squeeze()
                labels = labels[land_idx, :]
                #ax.scatter(output[:,0], output[:,1], linewidth=2, color='red')
                axs_flat[i].scatter(output[:,0], output[:,1], linewidth=2, color='c', s = 5)
                axs_flat[i].scatter(labels[:,0], labels[:,1], linewidth=2, color='m', s = 5)



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
    parser.add_argument("-imf", "--file_img",
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

    model.load_state_dict(torch.load("./models/" + args.model, map_location=torch.device('cpu')))
    print(evaluate(model, args.test_file, "cpu"))
    
    epoch = []
    error = []
    std = []
    # Setting up epoch, error and std arrays so that they can be outputted in a graph
    model_nopt = args.model.split(".pt")
    with open("./model_scores/" + model_nopt[0] + ".csv") as file:
        for line in file:
            scores = line.split(",")   
            epoch.append(float(scores[0]))
            error.append(float(scores[1]))
            std.append(float(scores[2]))
    
    file = open("./model_infos/" + model_nopt[0] + ".json")
    model_info = json.load(file)

    plt.plot(epoch, error)
    plt.xlabel("Iterations")
    plt.ylabel("Mean Error")
    plt.title('Error change during training')

    plt.scatter([model_info["iteration"]], [model_info["mean"]], color = 'red')
    plt.gca().legend(('Mean Error','Epoch of Saved Model'))
    plt.show()

    plt.plot(epoch, std)
    plt.xlabel("Epochs")
    plt.ylabel("Standard Deviation")
    plt.title('STD change during training')
    plt.scatter([model_info["iteration"]], [model_info["std"]], color = 'red')
    plt.gca().legend(('Standard Deviation','Epoch of Saved Model'))
    plt.show()

    if args.file_img != 'none':
        UTKFace = CustomImageDataset(args.file_img, 'UTKFace')
        train_dataloader = DataLoader(UTKFace, 
                                        batch_size=1, 
                                        shuffle=False)
        
        fig, axs = plt.subplots(2,5)
        axs_flat = axs.flatten()
        
        generate_images(train_dataloader, axs_flat)
        #plt.subplots_adjust(wspace=0, hspace=0)
        plt.legend(('Predicted output','Expected output'), loc="upper left")
        plt.show()

