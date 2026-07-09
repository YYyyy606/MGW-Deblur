import torch
import torch.nn as nn
import torch.nn.functional as F


class fftLoss(nn.Module):
    def __init__(self):
        super(fftLoss, self).__init__()

    def forward(self, x, y):
        diff = torch.fft.fft2(x) - torch.fft.fft2(y)  #diff是个元素数据类型是复数的张量
        loss = torch.sum(abs(diff))   #求解张量的模取和
        return loss