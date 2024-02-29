import torch
from torch import nn
from tqdm import tqdm

def train(model, train_loader, valid_loader, criterion, optimizer, device, print_every=10, epochs=20):
    for epoch in tqdm(range(epochs)):
        model.train()
        running_loss = 0.0

        for data in train_loader:
            X, y = data
            X, y = X.float().to(device), y.long().to(device)  

            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X.size(0)

        if (epoch + 1) % print_every == 0:
            avg_loss = running_loss / len(train_loader.dataset)
            val_accuracy = evaluate(model, valid_loader, device)
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_loader:
            X, y = data
            X, y = X.float().to(device), y.long().to(device)
            outputs = model(X)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    accuracy = correct / total
    return accuracy * 100
