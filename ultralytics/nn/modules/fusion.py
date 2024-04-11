import torch
import torch.nn as nn
from ultralytics.nn.modules import ShiftConv,GhostConv1

__all__ = ("FusionConv",)
class FusionConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        self.ShiftConv = ShiftConv(c1,c2)
        self.GhostConv1 = GhostConv1(c1,c2)
        self.cv1 = nn.Conv2d(c1,c2,k=1)
    def forward(self, x, y):
        x = self.ShiftConv(x)
        y = self.GhostConv1(y)
        return self.cv1(torch.cat([x,y],1)) 
        # return torch.cat([x,y],1)
    