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

import random

torch.manual_seed(42)
torch.use_deterministic_algorithms(True)

def evaluate(model):
    model.eval()

    # TODO: Choose best metric

    model.train()
    return 0

def train(model, train_loader, lr, momentum=0.9, epochs=5):
    loss_func = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)#, momentum=momentum)

    for epoch in range(epochs):
        for i, data in enumerate(train_loader, 0):
            images, _, _, _, landmarks = data   # images, age, gender, race, landmarks
            # Zero paramter gradients
            optimizer.zero_grad()

            outputs = model(images)
            loss = loss_func(outputs, landmarks[:, 31].float())
            loss.backward()
            optimizer.step()

            #sys.stdout.flush()
            sys.stdout.write(f"\rEpoch: {epoch}, Iteration: {i}, Loss: {loss}")
            #print_percent_done(i, 100)

            if i % 1000 == 0:
                print(f"Ep: {epoch}, iteration: {i}, loss: {loss.item()}")
    
    return model

if __name__ == "__main__":
    # Read in args
    parser = ArgumentParser()
    parser.add_argument("-f", "--file",
                        help="Path to data file.", 
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
                        default=0.001,
                        type=int,
                        metavar="INT")

    args = parser.parse_args()

    model = None
    if args.model == "convNN2":
        model = convNN2()

    UTKFace = CustomImageDataset(args.file, 'UTKFace')
    train_dataloader = DataLoader(UTKFace, batch_size=args.batch, shuffle=True)

    print(f"Training {args.model} from {args.file} with batch_size={args.batch}")

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(42)

    UTKFace = CustomImageDataset('landmark_list.txt', 'UTKFace')
    train_dataloader = DataLoader(UTKFace, 
                                    batch_size=64, 
                                    shuffle=True, 
                                    worker_init_fn=seed_worker,
                                    generator=g,)
    #console = curses.initscr()

    model = train(model, train_dataloader, args.learning_rate)