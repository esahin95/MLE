import torch 


def train(model, dataLoader, criterion, optimizer, epochs=1):
    '''
    Trains model for specified number of epochs
    '''
    # Get device
    device = next(model.parameters()).get_device()

    # training loop
    model.train()
    for epoch in range(epochs):
        print(f'Starting epoch: {epoch}\n')

        loss = 0.0
        for X, y in dataLoader:
              X, y = X.to(device), y.to(device)

              # prediction
              p = model(X)
              l = criterion(p, y)

              # backpropagation
              optimizer.zero_grad()
              l.backward()
              optimizer.step()

              # accumulate loss
              loss += l.item()
        loss /= len(dataLoader)
        print(f"Avg loss: {loss:>8f} \n")


def test(model, dataLoader, criterion):
    '''
    Evaluate mean test loss over given DataLoader
    '''
    # Get device
    device = next(model.parameters()).get_device()

    # Evaluate loss
    model.eval()
    with torch.no_grad():
        loss = 0.0
        for X, y in dataLoader:
            X, y = X.to(device), y.to(device)

            pred = model(X)
            loss += criterion(pred, y).item()
    loss /= len(dataLoader)
    
    return loss
    

def confusion(model, dataLoader):
    '''
    Compute confusion matrix and prediction accuracy for multi-class-classification
    '''
    # Get device
    device = next(model.parameters()).get_device()
    
    # number of classes
    model.eval()
    k = 0
    with torch.no_grad():
        for _, y in dataLoader:
            k = max(y.max(), k)
    k += 1

    # Evaluate Confusion matrix
    model.eval()
    C = torch.zeros((k,k))
    with torch.no_grad():
        for X, y in dataLoader:
            X, y = X.to(device), y.to(device)

            c = model.predict(X)
            for i in range(len(X)):
                C[c[i], y[i]] += 1

    # Accuracy
    a = torch.diag(C).sum() / C.sum()
    
    return C, a