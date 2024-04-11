import torch.nn as nn
import torchvision.models as models

class CNNModule(nn.Module):
    def __init__(self, in_channels, timesteps, n_classes):
        super(CNNModule, self).__init__()
        self.encoder = get_encoder(in_channels * timesteps)
        self.fc = nn.Linear(512, n_classes, bias=False)

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc(x)
        return x

def get_encoder(in_channels: int) -> nn.Module:
    resnet18 = models.resnet18(progress=True)

    resnet18.fc = nn.Identity()

    if in_channels != 3:
        resnet18.conv1 = nn.Conv2d(
            in_channels,
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )

    return resnet18
