import os, math, time, argparse, torch, torchvision, logging, pprint, warnings
import pandas as pd
from pandas import DataFrame
from pathlib import Path
import numpy as np
from tqdm import tqdm
from glob import glob
import matplotlib.pyplot as plt
from PIL import Image
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from prettytable import PrettyTable
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
from utils.utils import *
from models.WaveNet import WaveNet
from utils.dataset import WaveSpectraInfNPZ

warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

# Set up command line arguments
parser = argparse.ArgumentParser(description="Options for inference")

parser.add_argument("--model_path", action="store", type=str, required=True,
                    help="path to model and optimiser state_dict.pth files")
parser.add_argument("-d", "--data_path", action="store", type=str, required=True,
                    help="path to folder containing source images")
parser.add_argument("--device", type=str, default='cuda', choices=['cuda', 'cpu'],
                    help="device")

args = vars(parser.parse_args())

# Define paths and create directories
timestr = time.strftime("%m-%d_%H%M")

filepath = Path(args['model_path'])
inf_path = Path.joinpath(filepath.parents[1], 'inference', str(timestr))
filepath = filepath.parents[2]
inf_path.mkdir(parents=True, exist_ok=True)
model_path = Path(args['model_path'])

print("Arguments parsed")
for k, v in args.items():
    print("{:<15} {:<10}".format(str(k), str(v)))

print("\nCreated folders within referenced experiment: ")
print(filepath)
print(model_path)

# Device configuration
device = torch.device('cuda' if args['device'] == 'cuda' and torch.cuda.is_available() else 'cpu')
if device == 'cuda':
    torch.backends.cudnn.benchmark = True
print("\nDevice:\t" + str(device))

data_path = args['data_path']

with np.load(data_path) as test_data:
    data_files = test_data.files

# Create Dataset using custom class in utils.dataset.py
inf = WaveSpectraInfNPZ(data_path, data_files)

# Define DataLoader
inf_loader = DataLoader(inf, batch_size=1, shuffle=False)
inf_loader.requires_grad = False

print("==========================================================================================")
# Setup CNN using custom CAE class in models.model.py
network = WaveNet()
network.to(device)

print(summary(network, (1, 1, 29, 24), verbose=2))

for param in network.parameters():
    param.requires_grad = False

total_samples = len(inf)
n_iter = math.ceil(total_samples / 1)

print("Total samples: " + str(total_samples))

print('Setup complete!')
print("==========================================================================================")
print("==========================================================================================")


def predict():
    print("Generating predictions...")
    model = WaveNet()
    model.load_state_dict(torch.load(str(args['model_path']), map_location=torch.device(device)))
    print("Loading model from supplied model_path: {}".format(str(args['model_path']).replace("\\", '/')))
    model.to(device)
    model.eval()

    array_data = [None] * total_samples
    array_name = [None] * total_samples

    i = 0

    with tqdm(inf_loader, unit="sample") as tepoch:
        with torch.no_grad():
            for data, data_index in tepoch:
                data, data_min, data_max = min_max_01(data)

                data = data.to(device)
                tepoch.set_description(f"Inference")
                output = model(data)

                prediction = output.detach().cpu().numpy().squeeze()

                save_prediction(prediction,
                                data_index[0],
                                inf_path)

                array_data[i] = prediction
                array_name[i] = str(data_index).split(".")[0][2:7]

                i += 1

    data_dict = dict(zip(array_name, array_data))

    dict_keys = list(data_dict.keys())
    dict_keys.sort()
    sorted_data = {i: data_dict[i] for i in dict_keys}

    np.savez_compressed(str(inf_path) + "/predictions", **sorted_data)

    return None


predict()
