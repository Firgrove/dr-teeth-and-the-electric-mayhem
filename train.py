import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from net import convNN

def evaluate(model):
    model.eval()

    # TODO: Choose best metric

    model.train()
    return 0

def train(model, train_loader, lr=0.001, momentum=0.9, epochs=5):
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, momentum=momentum)

    for epoch in range(epochs):
        for i, (inputs, labels) in enumerate(train_loader, 0):
            
            # Zero paramter gradients
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            loss.backwards()
            optimizer.step()

            if i % 1000 == 0:
                print(f"Ep: {epoch}, iteration: {i}, loss: {loss.item()}")
    
    return model

if __name__ == "__main__":
    # TODO: We can add arguments here for different tasks. ie batch size
    batch_size = 32

    model = convNN()

    #TODO: Read in dataset
    trainLoader = DataLoader(dataSet, batch_size=batch_size, shuffle=True)

    model = train(model)