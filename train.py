import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

import os
import sys
import time
import curses

import numpy as np

from torch.utils.data import DataLoader
from argparse import ArgumentParser

import matplotlib.pyplot as plt

from dataset import CustomImageDataset
from net import convNN, convNN2, resnet18, resnet34, resnet50

from timer import Timer

# class Timer():
#     def __init__(self):
#         self.start_time = time.time()

#     def start(self):
#         self.start_time = time.time()

#     def elapsed_time(self):
#         current_time = time.time()

#         duration = current_time - self.start_time
    
#         hours = int(duration / 3600)
#         minutes = int((duration % 3600) / 60)
#         seconds = int((duration % 3600) % 60)

#         return f"{hours}h {minutes}m {seconds}s"

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


def train(model, train_loader, lr, device, valid_set, momentum=0.9, epochs=5, display_update_rate=100):
    """
    display_update_rate :: counts the number of iterations it takes before the program prints out an update
    """
    loss_func = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)#, momentum=momentum)

    timer = Timer()
    timer.start()

    for epoch in range(epochs):
        for i, data in enumerate(train_loader, 0):
            images, _, _, _, landmarks = data   # images, age, gender, race, landmarks
            # Zero paramter gradients
            optimizer.zero_grad()
            images, landmarks = images.to(device), landmarks.to(device)

            outputs = model(images)
            loss = loss_func(outputs, landmarks[:, 31])
            loss.backward() 
            optimizer.step()

            #sys.stdout.flush()
            #sys.stdout.write(f"\rEpoch: {epoch}, Iteration: {i}, Loss: {loss}")
            #print_percent_done(i, 100)

            sys.stdout.write(f"\r[{timer.elapsed_time()}] Epoch: {epoch}, Iteration: {i}, Loss: {loss}")
            sys.stdout.flush()
 
            if i % display_update_rate == 0:
                print(f"\r[{timer.elapsed_time()}] Epoch: {epoch}, Iteration: {i}, Loss: {loss}")
    
    print(f"\n\nTraining completed. Total Elapsed Time: {timer.elapsed_time()}")
    
    return model


def main():
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
    
    print("Using device: " + device)

    model = None
    if args.model == "convNN2":
        model = convNN2().to(device)
    elif args.model == "resnet18":
        model = resnet18().to(device)
    elif args.model == "resnet34":
        model = resnet34().to(device)
    elif args.model == "resnet50":
        model = resnet50().to(device)

    UTKFace = CustomImageDataset(args.train_file, 'UTKFace')
    train_dataloader = DataLoader(UTKFace, 
                                    batch_size=args.batch, 
                                    shuffle=True)

    print(f"Training {args.model} from {args.train_file} with batch_size={args.batch}\n")

    #console = curses.initscr()
    # Train model and then save it
    model = train(model, train_dataloader, args.learning_rate, device, args.validation_file, epochs=args.epochs)

    #TODO: Eval model here?


    # if models folder doesn't exist, create one
    if not os.path.isdir("models"):
        os.mkdir(os.getcwd() + "/models")

    # save model 
    model_path = f"models/{args.model}_batch{args.batch}_ep{args.epochs}_lr{args.learning_rate}_{args.train_file}.pt"
    torch.save(model.state_dict(), model_path)

if __name__ == "__main__":
    main()