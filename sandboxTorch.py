# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 10:39:23 2026

@author: emres
"""

import numpy as np
import matplotlib.pyplot as plt
from MLTools import *

import torch 
from torch import nn 
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

# Download training data from open datasets.
dataTrain = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
dataTest = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

# Create data loaders
bs = 64
dataLoaderTrain = DataLoader(dataTrain, batch_size=bs)
dataLoaderTest  = DataLoader(dataTest,  batch_size=bs)

for X, y in dataLoaderTest:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        self._pre = nn.Flatten()
        self._sequential = torch.nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
        self._post = nn.Identity()
    
    def forward(self, X):
        X = self._pre(X)
        X = self._sequential(X)
        X = self._post(X)
        return X
    
    def fit(self, dataLoader, epochs=5):
        numOfBatches = len(dataLoader)
        numOfExamples = len(dataLoader.dataset)
        
        # loss function
        lossFun = nn.CrossEntropyLoss()
        
        # optimizer
        optim = torch.optim.SGD(self.parameters(), lr=1e-3)
        
        # training loop
        for epoch in range(epochs):
            print(f'Current epoch: {epoch}\n')
            self.train()
            for X, y in dataLoader:
                X, y = X.to(device), y.to(device)
                
                # prediction
                pred = self(X)
                loss = lossFun(pred, y)
                
                # backpropagation
                optim.zero_grad()
                loss.backward()
                optim.step()
                
            self.eval()
            with torch.no_grad():
                loss, correct = 0.0, 0
                for X, y in dataLoader:
                    X, y = X.to(device), y.to(device)
                    
                    pred = model(X)
                    loss += lossFun(pred, y).item()
                    correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            loss /= numOfBatches
            correct /= numOfExamples
            print(f"Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {loss:>8f} \n")
            
            
            
            
    
model = NeuralNetwork().to(device)
print(model)

model.fit(dataLoaderTrain, epochs=5)