# src/ctr-gcn/model/ctrgc.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CTRGC(nn.Module):
    def __init__(self, in_c, out_c, V, reduction=4):
        super().__init__()
        self.V = V
        self.theta = nn.Conv2d(in_c, out_c, 1)
        self.A = nn.Parameter(torch.randn(V, V) * 1e-2)

        mid = max(1, out_c // reduction)
        self.emb1 = nn.Conv2d(out_c, mid, 1)
        self.emb2 = nn.Conv2d(out_c, mid, 1)
        self.proj = nn.Conv2d(mid, out_c, 1)

        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        y = self.theta(x)  # (N,C,T,V)

        q = self.emb1(y).mean(dim=2)   # (N,mid,V)
        k = self.emb2(y).mean(dim=2)   # (N,mid,V)
        corr = torch.einsum("ncv,ncw->nvw", q, k) / (q.shape[1] ** 0.5)
        corr = F.softmax(corr, dim=-1)

        A_hat = F.softmax(self.A + corr.mean(0), dim=-1)

        z = torch.einsum("nctv,vw->nctw", y, A_hat)
        return self.bn(z)
