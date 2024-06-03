import torch
import torch.nn.functional as F
from utils import choose_srf
import numpy as np


class SAMLoss(torch.nn.Module):
    def __init__(self):
        super(SAMLoss, self).__init__()

    def forward(self, y, gt):
        B, C, H, W = y.shape
        y_flat = y.reshape(B, C, -1)
        gt_flat = gt.reshape(B, C, -1)
        y_norm = torch.norm(y_flat, 2, dim=1)
        gt_norm = torch.norm(gt_flat, 2, dim=1)
        numerator = torch.sum(gt_flat*y_flat, dim=1)
        denominator = y_norm * gt_norm
        sam = torch.div(numerator, denominator + 1e-5)
        sam = torch.sum(torch.acos(sam)) / (B * H * W) * 180 / 3.14159
        return sam


class RGBLoss(torch.nn.Module):
    def __init__(self, rgb_type='Nikon'):
        super(RGBLoss, self).__init__()
        self.R = torch.from_numpy(np.array(np.float32(choose_srf(rgb_type)))).cuda()
        self.mse = torch.nn.MSELoss()

    def forward(self, x, y):
        rgb_recon = torch.einsum('mn,imjk->injk', self.R, x)
        out = self.mse(rgb_recon, y)
        return out


# from https://github.com/jxgu1016/Total_Variation_Loss.pytorch with slight modifications
class TVLoss(torch.nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]

class SparseLoss(torch.nn.Module):
    def __init__(self):
        super(SparseLoss, self).__init__()
    
    def forward(self, x):
        N, C, H, W = x.shape
        out = torch.sum(-x * torch.log(x+1e-7))
        return out / N / H / W
