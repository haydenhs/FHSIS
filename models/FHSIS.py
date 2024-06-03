import torch
import torch.nn as nn
import torch.nn.functional as F


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return torch.nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


class GuideNN(torch.nn.Module):
    def __init__(self, in_ch=3, n_feats=64, n_basis=4):
        super(GuideNN, self).__init__()
        self.conv1 = default_conv(in_channels=in_ch, out_channels=n_feats, kernel_size=3)
        self.conv2 = default_conv(in_channels=n_feats, out_channels=n_feats, kernel_size=3)
        self.conv3 = default_conv(in_channels=n_feats, out_channels=n_basis, kernel_size=3)
        self.relu = torch.nn.ReLU(True)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        y = self.relu(self.conv1(x))
        y = self.relu(self.conv2(y))
        y = self.softmax(self.conv3(y))
        return y


class ResBlock(torch.nn.Module):
    def __init__(self, conv, n_feats, kernel_size, bias=True, bn=False, act=torch.nn.ReLU(True), res_scale=1):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(torch.nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = torch.nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res


class Encoder(torch.nn.Module):
    def __init__(self, in_ch=3, out_ch=31, n_feats=128, n_blocks=3, n_basis=5, factor=4):
        super(Encoder, self).__init__()
        self.k = n_blocks + 1
        self.n = (in_ch + 1) * out_ch  # affine
        self.in_conv = default_conv(in_channels=out_ch, out_channels=n_feats, kernel_size=3)
        m = []
        for _ in range(n_blocks):
            m.append(ResBlock(default_conv, n_feats=n_feats, kernel_size=3, bn=False))
        self.convs = torch.nn.ModuleList(m)
        self.upsample = torch.nn.Upsample(scale_factor=factor, mode='nearest')
        self.relu = torch.nn.ReLU(True)
        self.out_conv1 = default_conv(in_channels=n_feats*self.k, out_channels=n_feats*self.k, kernel_size=3)
        self.out_conv2 = default_conv(in_channels=n_feats*self.k, out_channels=n_feats*self.k, kernel_size=1)
        self.out_conv3 = default_conv(in_channels=n_feats*self.k, out_channels=n_basis*self.n, kernel_size=1)

    def forward(self, x):
        y = self.in_conv(x)
        tmp = [y]
        for i in range(self.k-1):
            y = self.convs[i](y)
            tmp.append(y)
        y = torch.cat(tmp, 1)
        y = self.upsample(y)
        y = self.relu(self.out_conv1(y))
        y = self.relu(self.out_conv2(y))
        y = self.out_conv3(y)
        out = torch.stack(torch.split(y, self.n, 1),2)
        return out


class Gridup(torch.nn.Module):
    def __init__(self):
        super(Gridup, self).__init__()

    def forward(self, grid, guidemap): 
        # weight map: N, C, H, W
        # grid: N, K, C, h, w
        N, C, H, W = guidemap.shape
        h, w = grid.shape[3], grid.shape[4]
        f1, f2 = H//h, W//w
        guidemap = F.unfold(guidemap, kernel_size=(f1, f2), stride=(f1, f2))
        guidemap = guidemap.reshape(N, C, f1 * f2, h, w)
        out = torch.einsum('ikjmn,ijlmn->iklmn', grid, guidemap)
        out = out.reshape(N, -1, h*w)
        out = F.fold(out, output_size=(H, W), kernel_size=(f1, f2), stride=(f1, f2))
        return out # NxKxHxW


class ApplyCoeffs(nn.Module):
    def __init__(self, in_ch=3, out_ch=31):
        super(ApplyCoeffs, self).__init__()
        self.c_in = in_ch + 1
        self.c_out = out_ch

    def forward(self, coeff, full_res_input):
        N, _, H, W = full_res_input.shape
        device = full_res_input.get_device()
        aug = torch.ones(N, 1, H, W)
        if device >= 0:
            aug = aug.to(device)
        full_res_input = torch.cat((full_res_input, aug), dim=1)
        full_res_input = full_res_input.permute(0,2,3,1).unsqueeze(3)
        coeff = coeff.permute(0,2,3,1).reshape(N, H, W, self.c_in, self.c_out)
        out = torch.matmul(full_res_input, coeff)
        out = out.squeeze(3).permute(0,3,1,2)
        return out


class Net(torch.nn.Module):
    def __init__(self, in_ch=3, out_ch=31, n_basis=5, n_feats=128, factor=4):
        super(Net, self).__init__()
        self.guide = GuideNN(in_ch, n_feats, n_basis=n_basis)
        self.en = Encoder(in_ch=in_ch, out_ch=out_ch, n_basis=n_basis, n_feats=n_feats, factor=factor)
        self.slice = Gridup()
        self.apply_coeffs = ApplyCoeffs(in_ch, out_ch)
        self.pool = torch.nn.AvgPool2d(kernel_size=32, stride=32)

    def forward(self, lr, rgb, lr_rgb):
        wmap = self.guide(rgb)
        lr_grid = self.en(lr)
        hr_coeff = self.slice(lr_grid, wmap)
        lr_coeff = self.pool(hr_coeff)
        out = self.apply_coeffs(hr_coeff, rgb)
        out_lr = self.apply_coeffs(lr_coeff, lr_rgb)
        return out_lr, out


import numpy as np

def params_count(model):
    """
    Compute the number of parameters.
    Args:
        model (model): model to count the number of parameters.
    """
    return np.sum([p.numel() for p in model.parameters()]).item()

if __name__ == "__main__":
    net = Net()
    print(net)
    lr = torch.randn(1,31,16,16)
    rgb = torch.randn(1,3,512,512)
    lr_rgb = torch.randn(1,3,16,16)
    params = params_count(net)
    print(params/1024/1024)
    a, b = net(lr, rgb, lr_rgb)
    print(a.shape)
    print(b.shape)

