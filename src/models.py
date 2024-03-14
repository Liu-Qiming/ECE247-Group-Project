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

class EEGNet_Modified(nn.Module):
    '''
        hyperparameters s2:
        self, n_temporal_filters=8, 
        kernel_length=64, pool_size=8, 
        depth_multiplier=4, in_channels=22, dropout=0.3
    '''
    ''' test acc 0.74041
        self, in_samples=1000, n_temporal_filters=8, 
        kernel_length=64, pool_size=8,
        depth_multiplier=4, in_channels=22, dropout=0.4
    '''

    def __init__(
        self, in_samples=1000, n_temporal_filters=8, 
        kernel_length=64, pool_size=8,
        depth_multiplier=4, in_channels=22, dropout=0.4):

        super().__init__()

        self.input_shape = (in_channels, in_samples)
        kernel_length2 = 16
        Filter_Num_2 = depth_multiplier*n_temporal_filters

        self.temporal_conv1 = nn.Conv2d(1, n_temporal_filters, (1,kernel_length), padding='same', bias=False)
        self.batch_norm_1 = nn.BatchNorm2d(n_temporal_filters)
        self.depth_wise_conv = nn.Conv2d(n_temporal_filters, Filter_Num_2, (in_channels, 1), bias=False, groups=n_temporal_filters)
        self.batch_norm_2 = nn.BatchNorm2d(Filter_Num_2)
        self.elu = nn.ELU()
        self.average_pool1 = nn.AvgPool2d((1, pool_size), stride=(1, pool_size))
        self.average_pool2 = nn.AvgPool2d((1, pool_size), stride=(1, pool_size))
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.spatial_conv1 = nn.Conv2d(Filter_Num_2, Filter_Num_2, (1, kernel_length2), padding='same', bias=False)
        self.batch_norm_3 = nn.BatchNorm2d(Filter_Num_2)

        #NOTE: remove this if used as part of ATCNet, keep this if used as EGGNet
        self.temp_linear = nn.LazyLinear(4)

    def forward(self, x):
        # x should be (batch_size, 1, channels, time)
        h = x
        h = h.view(-1, 1, self.input_shape[0], self.input_shape[1])
        h = self.temporal_conv1(h)
        h = self.batch_norm_1(h)
        h = self.depth_wise_conv(h)
        h = self.batch_norm_2(h)
        h = self.elu(h)
        h = self.average_pool1(h)
        h = self.dropout1(h)
        h = self.spatial_conv1(h)
        h = self.batch_norm_3(h)
        h = self.elu(h)
        h = self.average_pool2(h)
        h = self.dropout2(h)

        #NOTE: remove this if used as part of ATCNet, keep this if used as EGGNet
        h=h.view(h.shape[0], -1)
        h=self.temp_linear(h)

        return h #(64, 32, 1, 15)
    
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
