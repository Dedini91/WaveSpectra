"""Custom dataset class"""
from PIL import Image
from torch.utils.data import Dataset


class WaveSpectra(Dataset):
    def __init__(self, data_paths, target_paths, transform):
        # super(Dataset, self).__init__()
        self.data_paths = data_paths

        # temp_path = self.data_paths[0].replace('\\', '/').split('/')[-1]
        # print(len(temp_path))
        self.target_paths = target_paths
        self.transform = transform

        self.img_paths = []
        self.ids = []

    def __getitem__(self, index):
        x_idx = self.data_paths[index].replace('\\', '/').split('/')[-1]
        y_idx = self.target_paths[index].replace('\\', '/').split('/')[-1]

        x = Image.open(self.data_paths[index])
        y = Image.open(self.target_paths[index])
        x = self.transform(x)
        y = self.transform(y)

        return x, y, x_idx, y_idx

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
