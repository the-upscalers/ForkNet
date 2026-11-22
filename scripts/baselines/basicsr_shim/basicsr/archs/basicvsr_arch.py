import torch.nn.functional as F
from torch import nn


class ResidualBlockNoBN(nn.Module):
    def __init__(self, num_feat=64, res_scale=1):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

    def forward(self, x):
        return x + self.conv2(F.relu(self.conv1(x))) * self.res_scale


class ConvResidualBlocks(nn.Module):
    """BasicSR's ConvResidualBlocks (parameter names main.0.*, main.2.<i>.conv{1,2}.*)."""

    def __init__(self, num_in_ch=3, num_out_ch=64, num_block=15):
        super().__init__()
        self.main = nn.Sequential(nn.Conv2d(num_in_ch, num_out_ch, 3, 1, 1, bias=True), nn.LeakyReLU(0.1, inplace=True),
                                  nn.Sequential(*[ResidualBlockNoBN(num_out_ch) for _ in range(num_block)]))

    def forward(self, fea):
        return self.main(fea)
