# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 16:34:12 2026

@author: emres
"""

import torch 
from torch import nn

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Download training data from open datasets.
dataset = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

device = torch.accelerator.current_accelerator().type
print(device)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        
        self._pre = nn.Flatten()
        self._sequential = nn.Sequential(
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
    
    def fit(self, data, epochs=5):
        pass
    
model = MLP()
print(model(dataset[0][0]))