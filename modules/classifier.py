import torch.nn as nn

class ClassificationModel(nn.Module):
    def __init__(self, encoder, feature_dim, num_classes=2):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        features = self.encoder(x)
        return self.classifier(features)
