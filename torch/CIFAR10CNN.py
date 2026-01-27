# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 10:39:23 2026

@author: emres
"""

# Import packages and name spaces
import torch 
from torch import nn 
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import Compose, ToTensor, Normalize 

from utils import train, test, confusion

# retrain
retrain = True
fname = "./tmp/CIFAR10CNN.pth"

# Input transformation
transform = Compose([
    ToTensor(), 
    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Download training data from open datasets.
dataTrain = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=transform
)

# Download test data from open datasets.
dataTest = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=transform
)

# Create data loaders
bs = 4
dataLoaderTrain = DataLoader(dataTrain, batch_size=bs, shuffle=True)
dataLoaderTest  = DataLoader(dataTest,  batch_size=bs, shuffle=False)

# Define machine learning model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        self._sequential = torch.nn.Sequential(
            nn.Conv2d(3, 6, 5), # input shape: (3, 32, 32)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )
    
    def forward(self, X):
        return self._sequential(X)

    def predict(self, X):
        return self(X).argmax(1)
    
# Set device to CUDA if available
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")
            
# Build model    
model = CNN().to(device)
if retrain:
    # Select loss function
    criterion = nn.CrossEntropyLoss()
    
    # Select optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    # Train model
    train(model, dataLoaderTrain, criterion, optimizer, epochs=10)

    # Save model
    torch.save(model.state_dict(), fname)
else:
    # Load model
    model.load_state_dict(torch.load(fname, weights_only=True))
print(model)

# number of parameters
N = 0
for p in model.parameters():
    N += torch.numel(p)
print(f"Number of parameters: {N}")

# Test model
loss = test(model, dataLoaderTest, nn.CrossEntropyLoss())
C, correct = confusion(model, dataLoaderTest)
print(f"Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {loss:>8f} \n")