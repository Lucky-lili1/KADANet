import torch.nn as nn
import torch.nn.functional as F
import torch

from module_util import CALayer


class FeedForward1(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward1, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.CA=CALayer(hidden_features,4)
        self.project_out = nn.Conv2d(hidden_features, dim//2, kernel_size=1, bias=bias)

    def forward(self, x):
        y=x
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x2=self.CA(x2)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x+y
