import torch
from torch import nn
import numpy as np

DROPOUT = 0.4
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=(3,3), stride=(1,1), padding=0),
            nn.BatchNorm2d(20, eps=1e-06, momentum=0.2, affine=True),  
            nn.ELU(),

            nn.Conv2d(20, 25, kernel_size=(3,3), stride=(1,1), padding=0),
            nn.BatchNorm2d(25, eps=1e-06, momentum=0.2, affine=True),  
            nn.ELU(),

            nn.Conv2d(25, 25, kernel_size=(3,3), stride=(1,1), padding=0),
            nn.BatchNorm2d(25, eps=1e-06, momentum=0.2, affine=True),  
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1,3), stride=(1,3)),
            nn.Dropout(DROPOUT),  
            
            nn.Conv2d(25, 22, kernel_size=(3,3), stride=(1,1), padding=0),
            nn.BatchNorm2d(22, eps=1e-06, momentum=0.2, affine=True),  
            nn.ELU(),
        )

        self.flattened_feature_nums = 101332
        
        self.fc = nn.Sequential(
            nn.Linear(self.flattened_feature_nums, 64),
            nn.BatchNorm1d(64, eps=1e-06, momentum=0.2, affine=True),  
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32, eps=1e-06, momentum=0.2, affine=True),  
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT),

            nn.Linear(32, 16),
            nn.BatchNorm1d(16, eps=1e-06, momentum=0.2, affine=True),  
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT),

            nn.Linear(16, 4)  
        )
        
        self._initialize_weights()

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    

class LSTMModel(nn.Module):
    def __init__(self, input_size=1000, hidden_size=128, output_size=4, num_layers=3):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers=self.num_layers, batch_first=True)
        self.bn1 = nn.BatchNorm1d(hidden_size)  
        
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers=self.num_layers, batch_first=True)
        self.bn2 = nn.BatchNorm1d(hidden_size)  
        
        self.lstm3 = nn.LSTM(hidden_size, hidden_size, num_layers=self.num_layers, batch_first=True)
        self.bn3 = nn.BatchNorm1d(hidden_size)  
        
        self.dropout = nn.Dropout(DROPOUT)
        
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm1(x, (h0, c0))
        out = self._permute(out)
        out = self.bn1(out)
        out = self._permute(out)
        
        out, _ = self.lstm2(out, (h0, c0))
        out = self._permute(out)
        out = self.bn2(out)
        out = self._permute(out)
        
        out = self.dropout(out)
        
        out, _ = self.lstm3(out, (h0, c0))
        out = self._permute(out)
        out = self.bn3(out)
        out = self._permute(out)
        
        out = self.fc(out[:, -1, :])
        
        return out
    
    def _permute(self, out):
        return out.permute(0, 2, 1)


class CNN_LSTM(nn.Module):
    def __init__(self):
        super(CNN_LSTM, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(22, 64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(64, eps=1e-06, momentum=0.2, affine=True),  
            nn.ELU(),

            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(128, eps=1e-06, momentum=0.2, affine=True),  
            nn.ELU(),

            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(128, eps=1e-06, momentum=0.2, affine=True),  
            nn.ELU(),
            nn.Dropout(DROPOUT),  

            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(128, eps=1e-06, momentum=0.2, affine=True),  
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(DROPOUT),  

            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(128, eps=1e-06, momentum=0.2, affine=True),  
            nn.ELU(),
            nn.Dropout(DROPOUT),  

            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(128, eps=1e-06, momentum=0.2, affine=True),  
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(DROPOUT),  

            nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(64, eps=1e-06, momentum=0.2, affine=True),  
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(DROPOUT),  
            
            nn.Conv1d(64, 22, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(22, eps=1e-06, momentum=0.2, affine=True),  
            nn.ELU(),
        )

        self.lstm = nn.LSTM(22, 64, 3, batch_first=True, dropout=DROPOUT)

        self.flattened_feature_nums = 22 * 120
        
        self.fc = nn.Sequential(
            nn.Linear(self.flattened_feature_nums, 512),
            nn.BatchNorm1d(512, eps=1e-06, momentum=0.2, affine=True),  
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256, eps=1e-06, momentum=0.2, affine=True),  
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128, eps=1e-06, momentum=0.2, affine=True),  
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT),

            nn.Linear(128, 4)  
        )
        
        self._initialize_weights()

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

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
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(module.bias, 0)
