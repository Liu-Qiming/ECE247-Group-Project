import torch
from torch import nn
from tqdm import tqdm

def train(model, train_loader, valid_loader, criterion, optimizer, device, print_every=10, epochs=20, patience=20):
    cur_val_max = float('inf')
    tol = 0

    for epoch in tqdm(range(epochs)):
        model.train()
        running_loss = 0.0
        for data in train_loader:
            X, y = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X.size(0)

        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for data in valid_loader:
                X, y = data[0].to(device), data[1].to(device)
                outputs = model(X)
                loss = criterion(outputs, y)
                val_loss += loss.item() * X.size(0)

        cur_val_mean_loss = val_loss / len(valid_loader.dataset)

        if (epoch + 1) % print_every == 0:
            avg_loss = running_loss / len(train_loader.dataset)
            val_accuracy = evaluate(model, valid_loader, device)
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

        if cur_val_mean_loss < cur_val_max:
            cur_val_max = cur_val_mean_loss
            tol = 0
        else:
            tol += 1
            if tol == patience:
                print('Early stopping!')
                break

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
