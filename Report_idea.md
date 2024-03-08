## Vanilla CNN
```python
        self.conv_layers = nn.Sequential(
            nn.Conv1d(22, 64, kernel_size=3, stride=1, padding=0),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=0),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(128, 22, kernel_size=3, stride=1, padding=0),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
```

Acc: 32.51%




## CNN with 8 CONV layers, and 3 linear FC. Both with batchnorm and dropout. train with 50 epoch

Acc: 54.85%

## CNN with a reduced layers and neurons. train on 20 epoch

```python
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
```
ACC: 45.6%

## VisionTransformer with one convlayer, one 2-layer tranformer and one projection head, train with 70 epoch
```python

    self.patch_embedding = nn.Conv2d(1, 64, kernel_size=(22,1), stride=(22,1))
    self.cls_token = nn.Parameter(torch.randn(1, 1, 64)*1)
    self.projection = nn.Linear(64, 4)
    self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=64, nhead=8), num_layers=2)
    self.positions = nn.Parameter(torch.randn(1001, 64))

optimizer = torch.optim.Adam(ult_cnn.parameters(), lr = 0.001, betas=(0.9, 0.99), eps=1e-6, weight_decay=0.0005)
```