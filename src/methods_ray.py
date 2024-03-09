import ray
from ray import tune
from ray import train as raytrain
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import torch
from torch import nn
from tqdm import tqdm
from src.models import *
from src.utils import *
from os import path
import os
USE_GPU = True
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
data_path = path.join(os.getcwd(), 'project_data/')

def train_ray_Ultimateconfig(config):
    # dataset
    X_test = np.load(data_path + "X_test.npy")
    y_test = np.load(data_path + "y_test.npy") - 769
    person_train_valid = np.load(data_path + "person_train_valid.npy")
    X_train_valid = np.load(data_path + "X_train_valid.npy")
    y_train_valid = np.load(data_path + "y_train_valid.npy") - 769
    person_test = np.load(data_path + "person_test.npy")
    X_train_valid = X_train_valid[:, np.newaxis, :, :]
    X_test = X_test[:, np.newaxis, :, :]
    X_train_valid = torch.from_numpy(X_train_valid).float()
    y_train_valid = torch.from_numpy(y_train_valid).long()
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).long()

    train_valid_dataset = TensorDataset(X_train_valid, y_train_valid)
    test_dataset = TensorDataset(X_test, y_test)

    # Splitting the dataset into train and valid sets
    num_train = int(0.8 * len(train_valid_dataset))
    num_valid = len(train_valid_dataset) - num_train
    train_indices, valid_indices = random_split(range(len(train_valid_dataset)), [num_train, num_valid])

    train_dataset = Subset(train_valid_dataset, train_indices)
    valid_dataset = Subset(train_valid_dataset, valid_indices)

    # Wrapping datasets with GaussianNoisyDataset
    train_dataset_noisy = GaussianNoisyDataset(train_dataset, mean=0., std=1.)
    valid_dataset_noisy = GaussianNoisyDataset(valid_dataset, mean=0., std=1.)
    test_dataset_noisy = GaussianNoisyDataset(test_dataset, mean=0., std=1.)

    batch_size = 32
    train_loader = DataLoader(train_dataset_noisy, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset_noisy, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset_noisy, batch_size=batch_size, shuffle=False)
    
    # model
    model = UltimateConvNet().to(device)
    
    # optimizer
    weight_decay = config['weight_decay'] if 'weight_decay' in config else 0.0005
    lr = config['lr'] if 'lr' in config else 0.001
    beta1 = config['beta1'] if 'beta1' in config else 0.9
    beta2 = config['beta2'] if 'beta2' in config else 0.99
    eps = config['eps'] if 'eps' in config else 1e-6
    
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay)
    
    criterion = nn.CrossEntropyLoss()

    for epoch in range(200):
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
        raytrain.report(metrics = {"loss" : cur_val_mean_loss})
        