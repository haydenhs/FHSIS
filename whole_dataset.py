import torch
from torch.utils.data import Dataset
from utils import choose_srf, downsample
import numpy as np
import h5py


def data_aug(label, mode=0):
    if mode == 0:
        # original
        return label
    elif mode == 1:
        # flip up and down
        return np.flipud(label)
    elif mode == 2:
        # rotate counterwise 90 degree
        return np.rot90(label)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        return np.flipud(np.rot90(label))
    elif mode == 4:
        # rotate 180 degree
        return np.rot90(label, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        return np.flipud(np.rot90(label, k=2))
    elif mode == 6:
        # rotate 270 degree
        return np.rot90(label, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        return np.flipud(np.rot90(label, k=3))


def to_tensor(data):
    return torch.from_numpy(data.copy()).permute(2, 0, 1)


class HSIDataset(Dataset):
    def __init__(self, input_dir, sr_factor=32, dataset_name='CAVE', lr_type='bicubic', rgb_type='Nikon', use_aug=True):
        if dataset_name not in ['CAVE', 'Harvard']:
            raise NotImplementedError
        with h5py.File(input_dir, 'r') as data:
            if dataset_name == 'CAVE':
                self.gt = np.float32(data['hsi'][...]) / 65535.
            elif dataset_name == 'Harvard':
                self.gt = np.float32(data['hsi'][...]) * 24
            else:
                self.gt = np.float32(data['hsi'][...])
        self.total = self.gt.shape[0]
        print('{} training set has a total of {} patches.'.format(dataset_name, self.total))
        self.sf = sr_factor
        self.mode = lr_type
        self.R = np.float32(choose_srf(rgb_type))
        self.aug = 8 if use_aug else 1

    def __len__(self):
        return self.total * self.aug

    def __getitem__(self, ind):
        patch_index = ind // self.aug
        aug_num = ind % self.aug if self.aug > 1 else 0
        gt_patch = self.gt[patch_index]
        rgb_patch = np.dot(gt_patch, self.R)
        # implement augmentation
        gt_patch, rgb_patch = data_aug(gt_patch, aug_num), data_aug(rgb_patch, aug_num)
        # spatial downsample
        lr_patch = np.float32(downsample(gt_patch, self.sf, self.mode))
        lr_rgb_patch = np.float32(downsample(rgb_patch, self.sf, self.mode))
        # trans to tensor
        lr_rgb_patch, gt_patch, rgb_patch, lr_patch = to_tensor(lr_rgb_patch),\
            to_tensor(gt_patch), to_tensor(rgb_patch), to_tensor(lr_patch)

        return lr_rgb_patch, lr_patch, rgb_patch, gt_patch


class HSITestset(Dataset):
    def __init__(self, input_dir, sr_factor=32, dataset_name='CAVE', lr_type='bicubic', rgb_type='Nikon'):
        if dataset_name not in ['CAVE', 'Harvard']:
            raise NotImplementedError
        with h5py.File(input_dir, 'r') as data:
            if dataset_name == 'CAVE':
                self.gt = np.float32(data['hsi'][...]) / 65535.
            elif dataset_name == 'Harvard':
                self.gt = np.float32(data['hsi'][...]) * 24
            else:
                self.gt = np.float32(data['hsi'][...])
        self.rgb = np.float32(np.dot(self.gt, choose_srf(rgb_type)))
        self.total = self.gt.shape[0]
        print('{} test set has a total of {} images.'.format(dataset_name, self.total))
        self.sf = sr_factor
        self.mode = lr_type

    def __len__(self):
        return self.total

    def __getitem__(self, ind):
        gt_patch = self.gt[ind]
        rgb_patch = self.rgb[ind]
        lr_patch = np.float32(downsample(gt_patch, self.sf, self.mode))
        lr_rgb_patch = np.float32(downsample(rgb_patch, self.sf, self.mode))
        # trans to tensor
        lr_rgb_patch, gt_patch, rgb_patch, lr_patch = to_tensor(lr_rgb_patch),\
            to_tensor(gt_patch), to_tensor(rgb_patch), to_tensor(lr_patch)
        return lr_rgb_patch, lr_patch, rgb_patch, gt_patch


if __name__ == "__main__":
    dataset = HSIDataset(input_dir='/home/CAVE_h5/CAVE_train.h5', sr_factor=32, dataset_name='CAVE', lr_type='wald', rgb_type='Nikon')
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1)
    for lr_rgb, lr, rgb, gt in loader:
        print(rgb.shape)
        print(gt.shape)
        print(lr.shape)
        print(lr_rgb.shape)
        break
