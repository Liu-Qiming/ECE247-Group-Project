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