import torch
from torch import nn
import numpy as np

DROPOUT = 0.5
DROPOUT_EEG = 0.4
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



class UltimateConvNet(nn.Module):
    def __init__(self, input_shape=(1, 22, 1000)):
        super(UltimateConvNet, self).__init__()
        self.input_shape = input_shape
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(1, 32), stride=(1, 1)),
            nn.BatchNorm2d(8),

            nn.Conv2d(8, 32, kernel_size=(22, 1), stride=(1, 1)),
            nn.BatchNorm2d(32),

            nn.ELU(),
            nn.AvgPool2d((1, 6), stride = (1, 6)),
            nn.Dropout(DROPOUT),

            nn.Conv2d(32, 32, kernel_size=(1, 16), stride=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ELU(),
            

            nn.AvgPool2d((1, 6), stride = (1, 6)),
            nn.Dropout(DROPOUT),
        )

        self.flatten = nn.Flatten()

        self.flattened_feature_nums = 768

        self.fc_layers = nn.Sequential(
            nn.Linear(self.flattened_feature_nums, 64),
            nn.ELU(),
            nn.BatchNorm1d(64),
            nn.Dropout(DROPOUT),
            
            nn.Linear(64, 32),
            nn.ELU(),
            nn.BatchNorm1d(32),
            nn.Dropout(DROPOUT),
            
            nn.Linear(32, 4)
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
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.kaiming_normal_(param.data, mode='fan_in', nonlinearity='relu')
                    elif 'weight_hh' in name:
                        nn.init.kaiming_normal_(param.data, mode='fan_in', nonlinearity='relu')
                    elif 'bias' in name:
                        param.data.fill_(0)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(module.bias, 0)

class RNN(nn.Module):
    def __init__(self, input_shape=(22, 1000), hidden_size=64, num_layers=2, n_classes=4, **kwargs):
        super(RNN, self).__init__()
        input_size, _ = input_shape
        self.hidden_size = hidden_size
        self.num_layer = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, **kwargs)
        self.fc = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1) # x is in batch, n_features, seq_len
        out, hn = self.rnn(x)  # (batch, seq_len, n_features)
        out = self.fc(out[:, -1, :]) # obtain the last output of the model
        return out

class EEGNet(nn.Module):
    def __init__(
        self, input_shape=(1, 22, 1000)):
        super(EEGNet, self).__init__()
        
        self.input_shape = input_shape

        super().__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(1,64), padding='zero'),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 32, kernel_size=(22, 1), groups=8),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d((1, 8), stride=(1, 8)),
            nn.Dropout(DROPOUT_EEG),
            nn.Conv2d(32, 32, kernel_size=(1, 16), padding='zero'),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d((1, 8), stride=(1, 8)),
            nn.Dropout(DROPOUT_EEG),
            nn.Flatten()
        )
        self.flattened_feature_nums = 480
        self.fc_layers = nn.Linear(self.flattened_feature_nums,4)

    def forward(self, x):
        h = self.conv_layers(x)
        h = self.fc_layers(h)

        return h
    
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
    
class ViT(nn.Module):
    def __init__(self, input_shape=(22, 1000), num_classes=4, num_head = 8, num_layers = 4, scaling = 1, patch_size = (2, 2), dim = 64):
        super(ViT, self).__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.num_head = num_head
        self.num_layers = num_layers
        self.scaling = scaling
        self.dim = dim
        self.seq_len = input_shape[0]*input_shape[1]//np.prod(patch_size) + 1
        self.patch_embedding = nn.Conv2d(1, self.dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim)*self.scaling)
        self.projection = nn.Linear(self.dim, self.num_classes)
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.dim, nhead=self.num_head), num_layers=self.num_layers)
        self.positions = nn.Parameter(torch.randn(self.seq_len, self.dim))
        
    def forward(self,x):
        x = self.patch_embedding(x)
        x = x.flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.positions
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = x[:, 0, :]
        x = self.projection(x)
        return x
