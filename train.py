import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from argparse import ArgumentParser

import matplotlib.pyplot as plt

from dataset import CustomImageDataset
from net import convNN

def evaluate(model):
    model.eval()

    # TODO: Choose best metric

    model.train()
    return 0

def train(model, train_loader, lr=0.001, momentum=0.9, epochs=5):
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)#, momentum=momentum)

    for epoch in range(epochs):
        for i, images, _, _, landmarks in enumerate(train_loader, 0):

            # Zero paramter gradients
            optimizer.zero_grad()

            outputs = model(images)
            loss = loss_func(outputs, landmarks[:, 31].float())
            loss.backward()
            optimizer.step()

            print(f"Epoch: {epoch}, Iteration: {i}, Loss: {loss}")

            if i % 1000 == 0:
                print(f"Ep: {epoch}, iteration: {i}, loss: {loss.item()}")
    
    return model

if __name__ == "__main__":
    parser = ArgumentParser()
    # Read in args
    parser.add_argument("-f", "--file",
                    help="Path to data file", metavar="FILE_PATH")
    parser.add_argument("-b", "--batch", 
                    help="Batch size for training", type=int, metavar="INT")

    # TODO: Parse args

    model = convNN()

    UTKFace = CustomImageDataset('landmark_list.txt', 'UTKFace')
    train_dataloader = DataLoader(UTKFace, batch_size=64, shuffle=True)

    model = train(model, train_dataloader)