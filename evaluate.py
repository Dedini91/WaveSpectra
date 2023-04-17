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
from utils.dataset import WaveSpectra

warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

# Set up command line arguments
parser = argparse.ArgumentParser(description="Options for model evaluation")

parser.add_argument("--verbose", action="store_true", default=False,
                    help="verbose output")
parser.add_argument("--model_path", action="store", type=str, required=True,
                    help="path to model and optimiser state_dict.pth files")
parser.add_argument("--img_path", action="store", type=str, required=True,
                    help="path to source images for evaluation and inference")
parser.add_argument("--target_path", action="store", type=str, required=True,
                    help="path to target images for evaluation and inference")
parser.add_argument("--device", type=str, default='cuda', choices=['cuda', 'cpu'],
                    help="device")
parser.add_argument("--errmaps", action='store_false', default=True,
                    help="turns off error maps in output files (may speed up evaluation)")
parser.add_argument("-l", "--loss", action="store", type=str, default='l1', choices=['l1', 'mse', 'huber'],
                    help="loss function")
parser.add_argument("--reduction", action="store", type=str, default='sum', choices=['mean', 'sum', 'none'],
                    help="reduction method")

args = vars(parser.parse_args())

# Define paths and create directories
timestr = time.strftime("%m-%d_%H%M")

filepath = Path(args['model_path'])
eval_path = Path.joinpath(filepath.parents[1], 'evaluation', str(timestr))
filepath = filepath.parents[2]
eval_path.mkdir(parents=True, exist_ok=True)
model_path = Path(args['model_path'])
metrics_path = Path.joinpath(eval_path, "metrics")
metrics_path.mkdir(parents=True, exist_ok=True)
logs_path = Path.joinpath(eval_path, "logs")
logs_path.mkdir(parents=True, exist_ok=True)

# Set up loggers/handlers
log = logging.getLogger('logger')
log.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(message)s')

fh = logging.FileHandler(str(logs_path) + '/log.log', mode='w', encoding='utf-8')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
log.addHandler(fh)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
log.addHandler(ch)

# Messages to stdout and stderr (file)
log.info("Arguments parsed")
if args['verbose']:
    for k, v in args.items():
        log.info("{:<15} {:<10}".format(str(k), str(v)))

log.info("\nCreated folders within referenced experiment: ")
if args['verbose']:
    log.info(filepath)
    log.info(logs_path)
    log.info(model_path)
    log.info(metrics_path)

log.info("==========================================================================================")
# Device configuration
device = torch.device('cuda' if args['device'] == 'cuda' and torch.cuda.is_available() else 'cpu')
if device == 'cuda':
    torch.backends.cudnn.benchmark = True
if args['verbose']:
    log.info("\nDevice:\t" + str(device))
print(args['img_path'])
print(args['target_path'])

# Get all image paths
x_test_paths = glob(str(args['img_path']) + '/*.jpg')
y_test_paths = glob(str(args['target_path']) + '/*.jpg')

# Create Dataset using custom class in utils.dataset.py
transformations = transforms.ToTensor()

test = WaveSpectra(x_test_paths, y_test_paths, transform=transformations)

# Define DataLoaders for train/val/test sets
test_loader = DataLoader(test, batch_size=1, shuffle=False)
test_loader.requires_grad = False

log.info("==========================================================================================")
# Setup CNN using custom CAE class in models.model.py
network = CAE()
network.to(device)
writer = SummaryWriter(str(eval_path) + "/logs/")

if args['verbose']:
    log.debug(summary(network, (1, 1, 64, 64), verbose=2))

for param in network.parameters():
    param.requires_grad = True

total_samples = len(test)
n_iter = math.ceil(total_samples / 1)

if args['verbose']:
    log.info("Total samples: " + str(total_samples))
    log.info("Batch size: " + str(1))
    log.info("# iterations: " + str(n_iter))

log.info("==========================================================================================")
# Set up loss function based on user-specified parameters
if args['loss'] == 'l1':
    loss_function = torch.nn.L1Loss(reduction=str(args['reduction']))
    torch.nn.L1Loss()
if args['loss'] == 'mse':
    loss_function = torch.nn.MSELoss(reduction=(str(args['reduction'])))
if args['loss'] == 'huber':
    loss_function = torch.nn.HuberLoss(reduction=str(args['reduction']))

log.info('Setup complete!')
log.info("==========================================================================================")
log.info("==========================================================================================")

image_id, dimensions, l1_sum, l1_mean, l2_sum, l2_mean, huber_sum, huber_mean = \
    [], [], [], [], [], [], [], [],


def evaluate():
    image_id.clear()
    l1_sum.clear()
    l1_mean.clear()
    l2_sum.clear()
    l2_mean.clear()
    huber_sum.clear()
    huber_mean.clear()
    log.info("Evaluating model...")
    model = CAE()
    model.load_state_dict(torch.load(str(args['model_path'])))
    if args['verbose']:
        log.info("Loading model from supplied model_path: {}".format(str(args['model_path']).replace("\\", '/')))

    model.to(device)
    model.eval()

    test_losses_l1, test_losses_mse, test_losses_huber = [], [], []

    with tqdm(test_loader, unit="batch") as tepoch:
        with torch.no_grad():
            i = 0
            for data, target, data_index, target_index in tepoch:
                pred_path = Path.joinpath(eval_path, str(data_index[0]).split('.')[0])
                pred_path.mkdir(parents=True, exist_ok=True)
                data = data.to(device)
                target = target.to(device)
                tepoch.set_description(f"Evaluating")
                output = model(data)

                output_min, output_max = output.min(), output.max()
                target_min, target_max = target.min(), target.max()
                output = (output - output_min) / (output_max - output_min) * (target_max - target_min) + target_min

                test_l1 = F.l1_loss(output, target, reduction='sum')
                test_mse = F.mse_loss(output, target, reduction='sum')
                test_huber = F.huber_loss(output, target, reduction='sum')

                if args['errmaps']:
                    plot_l1 = F.l1_loss(output, target, reduction='none').detach().cpu().numpy().squeeze()
                    plot_l2 = F.mse_loss(output, target, reduction='none').detach().cpu().numpy().squeeze()
                    plot_huber = F.huber_loss(output, target, reduction='none').detach().cpu().numpy().squeeze()

                prediction = output.detach().cpu().numpy().squeeze()

                save_eval(data.detach().cpu().numpy().squeeze(),
                          target.detach().cpu().numpy().squeeze(),
                          prediction,
                          torch.Tensor.tolist(test_l1),
                          torch.Tensor.tolist(test_mse),
                          torch.Tensor.tolist(test_huber),
                          data_index[0],
                          pred_path)
                i += 1

                if args['errmaps']:
                    save_error(target.detach().cpu().numpy().squeeze(),
                               prediction,
                               plot_l1,
                               plot_l2,
                               plot_huber,
                               data_index[0],
                               pred_path)

                image_id.append(str(data_index[0]))
                l1_sum.append(torch.Tensor.tolist(F.l1_loss(output, target, reduction='sum')))
                l1_mean.append(torch.Tensor.tolist(F.l1_loss(output, target, reduction='mean')))
                l2_sum.append(torch.Tensor.tolist(F.mse_loss(output, target, reduction='sum')))
                l2_mean.append(torch.Tensor.tolist(F.mse_loss(output, target, reduction='mean')))
                huber_sum.append(torch.Tensor.tolist(F.huber_loss(output, target, reduction='sum')))
                huber_mean.append(torch.Tensor.tolist(F.huber_loss(output, target, reduction='mean')))

                test_losses_l1.append(torch.Tensor.tolist(test_l1))
                test_losses_mse.append(torch.Tensor.tolist(test_mse))
                test_losses_huber.append(torch.Tensor.tolist(test_huber))

    return None


def losses_to_csv(name):
    loss_dict = {
        "Image ID": image_id,
        "L1/MAE (sum)": l1_sum,
        "L1/MAE (mean)": l1_mean,
        "L2/MSE (sum)": l2_sum,
        "L2/MSE (mean)": l2_mean,
        "Huber (sum)": huber_sum,
        "Huber (mean)": huber_mean
    }
    df = DataFrame.from_dict(loss_dict)
    df = df.sort_values(by='L1/MAE (sum)', ascending=True)
    log.info(name + ":")
    log.info(df)

    df.to_csv(str(metrics_path) + "/" + name + ".csv", sep=",", float_format="%.6f", header=True, index=False)


# Execute program
evaluate()
losses_to_csv('eval_results')
