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
from models.model import *
from utils.dataset import WaveSpectraInf

warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

# Set up command line arguments
parser = argparse.ArgumentParser(description="Options for inference")

parser.add_argument("--model_path", action="store", type=str, required=True,
                    help="path to model .pth file")
parser.add_argument("--img_path", action="store", type=str, required=True,
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

# Get all image paths
x_test_paths = glob(args['img_path'] + '/*.jpg')
y_test_paths = glob(args['img_path'] + '/*.jpg')

# Create Dataset using custom class in utils.dataset.py
transformations = transforms.ToTensor()

inf = WaveSpectraInf(x_test_paths, transform=transformations)

# Define DataLoaders for train/val/test sets
inf_loader = DataLoader(inf, batch_size=1, shuffle=False)
inf_loader.requires_grad = False

print("==========================================================================================")
# Setup CNN using custom CAE class in models.model.py
network = CAE()
network.to(device)
writer = SummaryWriter(str(inf_path) + "/logs/")

print(summary(network, (1, 1, 64, 64), verbose=2))

for param in network.parameters():
    param.requires_grad = True

total_samples = len(inf)
n_iter = math.ceil(total_samples / 1)

print("Total samples: " + str(total_samples))
print("Batch size: " + str(1))
print("# iterations: " + str(n_iter))

print('Setup complete!')
print("==========================================================================================")
print("==========================================================================================")


def predict():
    print("Evaluating model...")
    model = CAE()
    model.load_state_dict(torch.load(str(args['model_path'])))
    print("Loading model from supplied model_path: {}".format(str(args['model_path']).replace("\\", '/')))
    model.to(device)
    model.eval()

    with tqdm(inf_loader, unit="batch") as tepoch:
        with torch.no_grad():
            for data, data_index in tepoch:
                data = data.to(device)
                tepoch.set_description(f"Inference")
                output = model(data)

                output_min, output_max = output.min(), output.max()
                target_min, target_max = 0.01, 0.85
                output = (output - output_min) / (output_max - output_min) * (target_max - target_min) + target_min

                prediction = output.detach().cpu().numpy().squeeze()

                save_prediction(prediction,
                                data_index[0],
                                inf_path)

    return None


predict()
