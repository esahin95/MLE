import torch 

def move(f):
    def moved(model, *args, **kwargs):
        # Get device
        device = next(model.parameters()).get_device()

        # run at cpu and move back
        model = model.to("cpu")
        res = f(model, *args, **kwargs)
        model = model.to(device)

        return res
    return moved

def train(model, dataLoader, criterion, optimizer, epochs=1):
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


@move
def test(model, dataLoader, criterion):
    # Evaluate loss
    model.eval()
    with torch.no_grad():
        loss = 0.0
        for X, y in dataLoader:
            pred = model(X)
            loss += criterion(pred, y).item()
    loss /= len(dataLoader)
    
    return loss
    
@move
def confusion(model, dataLoader):
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
            c = model.predict(X)
            for i in range(len(X)):
                C[c[i], y[i]] += 1

    # Accuracy
    a = torch.diag(C).sum() / C.sum()
    
    return C, a