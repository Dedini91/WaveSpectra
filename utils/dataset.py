"""Custom dataset class"""
from PIL import Image
from torch.utils.data import Dataset
import torch


def pad_img(x):
    x = x.squeeze()
    pad_x = torch.ones((x.size(0)), 72, device=x.device, dtype=x.dtype) * 0.02
    pad_x[:, :x.size(1)] = x
    pad_x = torch.unsqueeze(pad_x, dim=0)
    return pad_x


class WaveSpectra(Dataset):
    def __init__(self, data_paths, target_paths, transform):
        self.data_paths = data_paths

        self.target_paths = target_paths
        self.transform = transform

        self.img_paths = []
        self.ids = []

    def __getitem__(self, index):
        x_idx = self.data_paths[index].replace('\\', '/').split('/')[-1]
        y_idx = self.target_paths[index].replace('\\', '/').split('/')[-1]

        x = Image.open(self.data_paths[index])
        y = Image.open(self.target_paths[index])
        x_orig = self.transform(x)
        x = pad_img(x_orig)
        y = self.transform(y)

        return x, y, x_idx, y_idx, x_orig

    def __len__(self):
        return len(self.data_paths)


class WaveSpectraInf(Dataset):
    def __init__(self, data_paths, transform):
        self.data_paths = data_paths
        self.transform = transform
        self.img_paths = []
        self.ids = []

    def __getitem__(self, index):
        x_idx = self.data_paths[index].replace('\\', '/').split('/')[-1]
        x = Image.open(self.data_paths[index])
        x = self.transform(x)

        return x, x_idx

    def __len__(self):
        return len(self.data_paths)
