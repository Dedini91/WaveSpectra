"""Custom dataset class"""
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch


def pad_img(x):
    x = x.squeeze()
    pad_x = torch.zeros((x.size(0)), 72, device=x.device, dtype=x.dtype)
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


class WaveSpectraNPZ(Dataset):
    def __init__(self, x_path, y_path, x_files, y_files):
        self.x_path = x_path
        self.y_path = y_path
        self.x_files = x_files
        self.y_files = y_files

    def __getitem__(self, idx):
        x = np.load(str(self.x_path))[str(self.x_files[idx]).zfill(5)].astype(np.float32)
        y = np.load(str(self.y_path))[str(self.y_files[idx]).zfill(5)].astype(np.float32)

        x_orig = torch.from_numpy(x)
        x = pad_img(x_orig)
        y = torch.from_numpy(y)
        y = torch.unsqueeze(y, dim=0)

        return x, y, str(self.x_files[idx]).zfill(5), x_orig

    def __len__(self):
        return len(np.load(self.x_path).files)


class WaveSpectraInfNPZ(Dataset):
    def __init__(self, x_path, x_files):
        self.x_path = x_path
        self.x_files = x_files

    def __getitem__(self, idx):
        x = np.load(str(self.x_path))[str(self.x_files[idx]).zfill(5)].astype(np.float32)

        x_orig = torch.from_numpy(x)
        x = pad_img(x_orig)

        return x, str(self.x_files[idx]).zfill(5)

    def __len__(self):
        return len(np.load(self.x_path).files)


class WaveSpectraNpzRAM(Dataset):
    def __init__(self, x_path, y_path, device):
        self.x_data = np.load(x_path)
        self.x_files = self.x_data.files

        self.y_data = np.load(y_path)
        self.y_files = self.y_data.files

        self.device = device

    def __getitem__(self, idx):
        x = self.x_data[str(self.x_files[idx]).zfill(5)]
        y = self.y_data[str(self.y_files[idx]).zfill(5)]

        x_orig = torch.from_numpy(x)
        x = pad_img(x_orig).to(self.device)
        y = torch.from_numpy(y).to(self.device)
        y = torch.unsqueeze(y, dim=0)

        return x, y, str(self.x_files[idx]).zfill(5), x_orig

    def __len__(self):
        return len(self.x_files)


class WaveSpectraInfNpzRAM(Dataset):
    def __init__(self, x_path, device):
        self.x_data = np.load(x_path)
        self.x_files = self.x_data.files
        self.device = device

    def __getitem__(self, idx):
        x = self.x_data[str(self.x_files[idx]).zfill(5)]

        x_orig = torch.from_numpy(x)
        x = pad_img(x_orig).to(self.device)

        return x, str(self.x_files[idx]).zfill(5)

    def __len__(self):
        return len(self.x_files)
