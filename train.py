import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

import sys
import time
import curses

import numpy as np

from torch.utils.data import DataLoader
from argparse import ArgumentParser

import matplotlib.pyplot as plt

from dataset import CustomImageDataset
from net import convNN, convNN2

def print_percent_done(index, total, bar_len=50, title='Please wait'):
    '''
    index is expected to be 0 based index. 
    0 <= index < total
    '''
    percent_done = (index+1)/total*100
    percent_done = round(percent_done, 1)

    done = round(percent_done/(100/bar_len))
    togo = bar_len-done

    done_str = '█'*int(done)
    togo_str = '░'*int(togo)

    console.clear()
    console.addstr(f'Loss so far')
    console.addstr(f'\t⏳{title}: [{done_str}{togo_str}] {percent_done}% done')
    console.refresh()

    if round(percent_done) == 100:
        print('\t✅')

torch.manual_seed(42)

def evaluate(model, valid_set_path, device):
    UTKFace = CustomImageDataset(valid_set_path, 'UTKFace')
    valid_set = DataLoader(UTKFace, 
                            500, 
                            shuffle=True)

    # We're calculating the distance ourselves as using MSE loss doesnt 
    # allow us to square root terms individually.
    model.eval()
    with torch.no_grad():
        for images, _, _, _, landmarks in valid_set:
            images, landmarks = images.to(device), landmarks.to(device)
            outputs = model(images)

            difference = torch.square(outputs - landmarks[:, 31]).to(device)
            difference = torch.sqrt(difference[:, 0] + difference[:, 1])

    model.train()
    return torch.mean(difference).item(), torch.std(difference).item()

def train(model, train_loader, lr, device, valid_set, momentum=0.9, epochs=5):
    loss_func = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)#, momentum=momentum)

    for epoch in range(epochs):
        for i, data in enumerate(train_loader, 0):
            images, _, _, _, landmarks = data   # images, age, gender, race, landmarks
            # Zero paramter gradients

            print(images.shape)

            optimizer.zero_grad()
            images, landmarks = images.to(device), landmarks.to(device)

            outputs = model(images)
            loss = loss_func(outputs, landmarks[:, 31])
            loss.backward()
            optimizer.step()

            #sys.stdout.flush()
            sys.stdout.write(f"\rEpoch: {epoch}, Iteration: {i},  Loss: {loss}")
            #, Score: {evaluate(model, valid_set, device)# }")
            #print_percent_done(i, 100)

            if i % 1000 == 0:
                print(f"Ep: {epoch}, iteration: {i}, loss: {loss.item()}")
    
    return model

if __name__ == "__main__":
    # Read in args
    parser = ArgumentParser()
    parser.add_argument("-f", "--train_file",
                        help="Path to data file.", 
                        metavar="FILE_PATH", 
                        default="landmark_list.txt")
    parser.add_argument("-vf", "--validation_file",
                        help="Choose file to use for validation.",
                        metavar="FILE_PATH",
                        default="landmark_list.txt")
    parser.add_argument("-b", "--batch", 
                        help="Batch size for training.", 
                        type=int, 
                        metavar="INT",
                        default=64)
    parser.add_argument("-m", "--model",
                        help="Choose which model structure to use.",
                        default="convNN2",
                        metavar="MODEL_NAME")
    parser.add_argument("-lr", "--learning_rate",
                        help="Learning rate to run the optimizer function with.",
                        default=0.0001,
                        type=float,
                        metavar="FLOAT")
    parser.add_argument("--cuda",
                        help="Add this argument to run the code using GPU acceleration.",
                        action="store_true")
    parser.add_argument("-e", "--epochs",
                        help="Dictate number of epochs to train for.",
                        type=int,
                        metavar="INT",
                        default=5)

    args = parser.parse_args()

    device = "cpu"
    if args.cuda and torch.cuda.is_available():
        device = "cuda"

    model = None
    if args.model == "convNN2":
        model = convNN2().to(device)

    UTKFace = CustomImageDataset(args.train_file, 'UTKFace')
    train_dataloader = DataLoader(UTKFace, 
                                    batch_size=args.batch, 
                                    shuffle=True)

    print(f"Training {args.model} from {args.train_file} with batch_size={args.batch}")

    #console = curses.initscr()

    # Train model and then save it
    model = train(model, train_dataloader, args.learning_rate, device, args.validation_file, epochs=args.epochs)

    #TODO: Eval model here?
    model_path = f"./models/{args.model}_{args.train_file}.pt"
    torch.save(model.state_dict(), model_path)
