import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 112x112
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 56x56
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 28x28
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)),  # 1x1
            nn.Flatten()
        )
        self.feature_dim = 256  # After AdaptiveAvgPool2d(1,1) and flatten

    def forward(self, x):
        return self.network(x)
