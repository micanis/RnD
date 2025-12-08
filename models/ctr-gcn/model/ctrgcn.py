# src/ctr-gcn/model/ctrgcn.py
import torch
import torch.nn as nn
from model.ctrgc import CTRGC

class CTRGCN(nn.Module):
    def __init__(self, V=25, num_class=60, in_channels=3, base_channels=64):
        super().__init__()
        self.data_bn = nn.BatchNorm1d(in_channels * V)

        self.g1 = CTRGC(in_channels, base_channels, V)
        self.g2 = CTRGC(base_channels, base_channels, V)
        self.g3 = CTRGC(base_channels, base_channels, V)

        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(base_channels, num_class)

    def forward(self, x):  # x: (N,C,T,V)
        N,C,T,V = x.shape
        x = x.permute(0,1,3,2).reshape(N, C*V, T)
        x = self.data_bn(x).reshape(N, C, V, T).permute(0,1,3,2)

        y = self.g1(x)
        y = self.g2(y)
        y = self.g3(y)

        y = self.pool(y).reshape(N, -1)
        return self.fc(y)
