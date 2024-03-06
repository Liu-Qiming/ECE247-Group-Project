import torch
from torch import nn
import numpy as np

DROPOUT = 0.5
class ConvNet(nn.Module):
    def __init__(self, input_shape=(1, 22, 1000)):
        super(ConvNet, self).__init__()
        self.input_shape = input_shape
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=(1, 25), stride=(1, 1)),
            nn.ELU(),
            nn.BatchNorm2d(20),

            nn.Conv2d(20, 20, kernel_size=(22, 1), stride=(1, 1)),
            nn.ELU(),
            nn.BatchNorm2d(20),

            nn.Conv2d(20, 25, kernel_size=(1, 10), stride=(1, 1)),
            nn.ELU(),
            nn.BatchNorm2d(25),
            nn.MaxPool2d(kernel_size=(1, 3)),

            nn.Conv2d(25, 50, kernel_size=(1, 10), stride=(1, 1)),
            nn.ELU(),
            nn.BatchNorm2d(50),
            nn.MaxPool2d(kernel_size=(1, 3)),

            nn.Conv2d(50, 100, kernel_size=(1, 10), stride=(1, 1)),
            nn.ELU(),
            nn.BatchNorm2d(100),
            nn.MaxPool2d(kernel_size=(1, 3)),
            nn.Dropout(DROPOUT),
        )

        self.flatten = nn.Flatten()

        self.flattened_feature_nums = 3100

        self.fc_layers = nn.Sequential(
            nn.Linear(self.flattened_feature_nums, 100),
            nn.ELU(),
            nn.BatchNorm1d(100),
            nn.Dropout(DROPOUT),

            nn.Linear(100, 4)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        return x

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        
        self.lstm = nn.Sequential(
            nn.LSTM(22, 64, 3, batch_first = True, dropout=DROPOUT)
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(64, 4)
        )
    
    def forward(self, x):
        x = x.squeeze(1)
        x = x.permute(0,2,1)
        x, _ = self.lstm(x)
        x = self.fc_layers(x[:, -1, :])
        return x
        




class CNN_LSTM(nn.Module):
    def __init__(self):
        super(CNN_LSTM, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(22, 25, kernel_size=10),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=3),
            nn.BatchNorm1d(25),

            nn.Conv1d(25, 30, kernel_size=12),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=3),
            nn.BatchNorm1d(30),
            nn.Dropout(DROPOUT),

            nn.Conv1d(30, 50, kernel_size=12),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=3),
            nn.BatchNorm1d(50),
            nn.Dropout(DROPOUT),
        )
        self.lstm = nn.LSTM(input_size=50, hidden_size=64, num_layers=3, batch_first=True)
        self.fc_layers = nn.LazyLinear(4)

    def forward(self, x):
        x = x.squeeze(1)
        x = self.conv_layers(x)
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        out = self.fc_layers(out[:, -1, :])
        return out

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.kaiming_normal_(param.data, mode='fan_in', nonlinearity='relu')
                    elif 'weight_hh' in name:
                        nn.init.kaiming_normal_(param.data, mode='fan_in', nonlinearity='relu')
                    elif 'bias' in name:
                        param.data.fill_(0)
            elif isinstance(module, nn.LazyLinear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(module.bias, 0)
