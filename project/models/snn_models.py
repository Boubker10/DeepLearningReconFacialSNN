from project.models import sew_resnet
from spikingjelly.clock_driven import neuron, functional, surrogate
import torch.nn as nn


class SNNModule(nn.Module):
    def __init__(self, in_channels, timesteps, n_classes):
        super(SNNModule, self).__init__()
        self.encoder = get_encoder_snn(in_channels, timesteps)
        out_encoder = 512 # sew_resnet_18

        self.fc = nn.Linear(out_encoder, n_classes, bias=False)

    def forward(self, x):
        x = x.permute(1, 0, 2, 3, 4)
        # IMPORTANT: always apply reset_net before a new forward
        functional.reset_net(self.encoder)
        functional.reset_net(self.fc)

        x = self.encoder(x)
        x = self.fc(x)

        return x


def get_encoder_snn(in_channels: int, T: int):
    resnet = sew_resnet.MultiStepSEWResNet(
        block=sew_resnet.MultiStepBasicBlock,
        layers=[2, 2, 2, 2],
        zero_init_residual=True,
        T=T,
        cnf="ADD",
        multi_step_neuron=neuron.MultiStepIFNode,
        detach_reset=True,
        surrogate_function=surrogate.ATan(),
    )

    if in_channels != 3:
        resnet.conv1 = nn.Conv2d(
            in_channels,
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )

    return resnet

