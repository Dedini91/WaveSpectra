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
from utils.utils import *
from models.WaveNet import WaveNet
from utils.dataset import WaveSpectraNPZ
from torchmetrics import StructuralSimilarityIndexMeasure

warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

# Set up command line arguments
parser = argparse.ArgumentParser(description="Options for model evaluation")

parser.add_argument("--verbose", action="store_true", default=False,
                    help="verbose output")
parser.add_argument("--model_path", action="store", type=str, required=True,
                    help="path to checkpoint")
parser.add_argument("-d", "--data", action="store", type=str, default='data', required=True,
                    help="path to directory containing .npz files")
parser.add_argument("--device", type=str, default='cuda', choices=['cuda', 'cpu'],
                    help="device")
parser.add_argument("--save_preds", action="store_false", default=True,
                    help="pass argument to prevent saving of predictions")

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

x_test_path = "data/x_test.npz"
y_test_path = "data/y_test.npz"

with np.load(x_test_path) as x_test_data:
    x_test_data_files = x_test_data.files

with np.load(y_test_path) as y_test_data:
    y_test_data_files = y_test_data.files

# Create Dataset using custom class in utils.dataset.py
test = WaveSpectraNPZ(x_test_path, y_test_path, x_test_data_files, y_test_data_files)

# Define DataLoaders for train/val/test sets
test_loader = DataLoader(test, batch_size=1, shuffle=False)
test_loader.requires_grad = False

log.info("==========================================================================================")
# Model setup
model = WaveNet()
model.to(device)

if args['verbose']:
    log.debug(summary(model, (1, 1, 29, 24), verbose=2))

for param in model.parameters():
    param.requires_grad = False

total_samples = len(test)
n_iter = math.ceil(total_samples / 1)

if args['verbose']:
    log.info("Total samples: " + str(total_samples))

log.info("==========================================================================================")


def compute_ssim(output, target):
    ssim_similarity = ssim(output, target)
    ssim_loss = 1 - ssim_similarity
    return ssim_similarity, ssim_loss


ssim = StructuralSimilarityIndexMeasure(reduction='elementwise_mean').to(device)
l1 = torch.nn.L1Loss(reduction='sum').to(device)

log.info('Setup complete!')
log.info("==========================================================================================")
log.info("==========================================================================================")

ssimLoss, ssimSim = [], []
image_id, dimensions, l1_sum = [], [], []


def evaluate():
    log.info("Evaluating model...")

    model.load_state_dict(torch.load(str(args['model_path'])))
    if args['verbose']:
        log.info("Loading model from supplied model_path: {}".format(str(args['model_path']).replace("\\", '/')))

    model.to(device)
    model.eval()

    with tqdm(test_loader, unit="sample") as tepoch:
        with torch.no_grad():
            i = 0
            for data, target, data_index, target_index, data_orig in tepoch:
                pred_path = Path.joinpath(eval_path, str(data_index[0]).split('.')[0])
                pred_path.mkdir(parents=True, exist_ok=True)

                data, data_min, data_max = min_max_01(data)
                target, target_min, target_max = min_max_01(target)

                data = data.to(device)
                target = target.to(device)
                tepoch.set_description(f"Evaluating")
                output = model(data)

                # Compute SSIM
                ssim_similarity, ssim_loss = compute_ssim(output, target)

                # Compute l1 loss
                l1_loss = l1(output, target)

                l1_sum.append(l1_loss.item())
                ssimLoss.append(ssim_loss.item())
                ssimSim.append(ssim_similarity.item())

                data = min_max_revert(data, data_min, data_max)
                target = min_max_revert(target, target_min, target_max)
                output = min_max_revert(output, target_min, target_max)

                prediction = output.detach().cpu().numpy().squeeze()

                if args['save_preds']:
                    save_eval(data_orig.detach().cpu().numpy().squeeze(),
                              target.detach().cpu().numpy().squeeze(),
                              prediction,
                              l1_sum[-1],
                              ssimLoss[-1],
                              data_index[0],
                              pred_path)

                i += 1

                image_id.append(str(data_index[0]))

            results = PrettyTable(["L1 Loss",
                                   '|',
                                   "SSIM Loss",
                                   "SSIM Similarity"])
            results.add_rows([['{:.4f}'.format(sum(l1_sum) / len(l1_sum)),
                               '|',
                               '{:.4f}'.format(sum(ssimLoss) / len(ssimLoss)),
                               '{:.4f}'.format(sum(ssimSim) / len(ssimSim))]])

            results.border = False
            log.info(results)
            log.info("\n")

    return None


def losses_to_csv(name):
    loss_dict = {
        "Image ID": image_id,
        "L1/MAE": l1_sum,
        "SSIM error": ssimLoss,
        "SSIM similarity": ssimSim
    }
    df = DataFrame.from_dict(loss_dict)
    df = df.sort_values(by='L1/MAE', ascending=True)
    log.info(name + ":")
    log.info(df)

    df.to_csv(str(metrics_path) + "/" + name + ".csv", sep=",", float_format="%.6f", header=True, index=False)
    log.info("Saved as " + str(metrics_path) + "/" + name + ".csv")


# Execute program
evaluate()
losses_to_csv('eval_results')
