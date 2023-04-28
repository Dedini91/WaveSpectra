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
from utils.ssim import SSIM

from models.model import *
from utils.dataset import WaveSpectra
from utils.utils import *

warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

# Set up command line arguments
parser = argparse.ArgumentParser(description="Options for training models")

parser.add_argument("--verbose", action="store_true", default=False,
                    help="verbose output")
parser.add_argument("-n", "--name", action="store", type=str, required=True,
                    help="experiment name")
parser.add_argument("-d", "--data", action="store", type=str, required=True,
                    help="path to processed dataset")
parser.add_argument("--prototyping", action="store_true", default=False,
                    help="prototyping mode (train on reduced dataset)")
parser.add_argument("--track", action='store', type=str, required=False,
                    help="saves all predictions for a specified source image (file name only e.g. 00178)")
parser.add_argument("--model_path", action="store", type=str, required=False,
                    help="path to saved model and optimiser .pth files")
parser.add_argument("--device", type=str, default='cuda', choices=['cuda', 'cpu'],
                    help="device")
parser.add_argument("-e", "--num-epochs", type=int, default=100,
                    help="number of epochs")
parser.add_argument("-b", "--batch-size", type=int, default=1,
                    help="batch size")
parser.add_argument("--lr", "--learning_rate", type=float, default=0.00005,
                    help="learning rate")
parser.add_argument("-o", "--optimizer", action="store", type=str, default='adamw', choices=['adam', 'adamw', 'SGD'],
                    help="optimizer")
parser.add_argument("-l", "--loss", action="store", type=str, default='ssim', 
                    choices=['l1', 'mse', 'huber', 'cosine', 'ssim'],
                    help="loss function")
parser.add_argument("--reduction", action="store", type=str, default='sum', choices=['mean', 'sum', 'none'],
                    help="reduction method")
parser.add_argument("-m", "--momentum", type=float, default=0.9,
                    help="momentum for SGD, beta1 for adam")
parser.add_argument("--scheduler", action="store_false", default=True,
                    help="learning rate scheduler - cosine annealing (pass argument to disable)")
parser.add_argument("--lr_min", action="store", type=float, default=0.00001,
                    help="minimum learning rate for scheduler")
parser.add_argument("--decay", type=float, default=0.0,
                    help="weight decay rate (default off)")
parser.add_argument("--interval", type=int, default=1,
                    help="model checkpoint interval (epochs)")

args = vars(parser.parse_args())

# Define paths and create directories
timestr = time.strftime("%m-%d_%H%M")

filepath = Path('results', str(args['name']), str(timestr))
filepath.mkdir(parents=True, exist_ok=True)

logs_path = Path.joinpath(filepath, "logs")
model_path = Path.joinpath(filepath, "model")
metrics_path = Path.joinpath(filepath, "metrics")
preds_tr_path = Path.joinpath(filepath, "predictions", "training")
preds_val_path = Path.joinpath(filepath, "predictions", "validation")
preds_eval_path = Path.joinpath(filepath, "predictions", "evaluation")

logs_path.mkdir(parents=True, exist_ok=True)
model_path.mkdir(parents=True, exist_ok=True)
metrics_path.mkdir(parents=True, exist_ok=True)
preds_tr_path.mkdir(parents=True, exist_ok=True)
preds_val_path.mkdir(parents=True, exist_ok=True)
preds_eval_path.mkdir(parents=True, exist_ok=True)


# Set up loggers/handlers
log = logging.getLogger('logger')
log.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(message)s')

fh = logging.FileHandler(str(logs_path) + '/log.log', mode='w', encoding='utf-8')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
log.addHandler(fh)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)  # Prints to both log and CLI
ch.setFormatter(formatter)
log.addHandler(ch)

# Messages to stdout and stderr (file)
log.info("Arguments parsed - Current configuration:")
if args['verbose']:
    for k, v in args.items():
        log.info("{:<15} {:<10}".format(str(k), str(v)))

log.info("\nCreated folder structure for experiment: ")
if args['verbose']:
    log.info(filepath)
    log.info(logs_path)
    log.info(model_path)
    log.info(metrics_path)
    log.info(preds_tr_path)
    log.info(preds_val_path)
    log.info(preds_eval_path)

log.info("==========================================================================================")
# Device configuration
device = torch.device('cuda' if args['device'] == 'cuda' and torch.cuda.is_available() else 'cpu')
if device == 'cuda':
    torch.backends.cudnn.benchmark = True
if args['verbose']:
    log.info("\nDevice:\t" + str(device))

# Get all image paths
if args["prototyping"]:  # Reduced for testing
    x_train_paths = glob(args['data'] + '/red_x_train/*.jpg')
    y_train_paths = glob(args['data'] + '/red_y_train/*.jpg')
    x_val_paths = glob(args['data'] + '/red_x_val/*.jpg')
    y_val_paths = glob(args['data'] + '/red_y_val/*.jpg')
    x_test_paths = glob(args['data'] + '/red_x_test/*.jpg')
    y_test_paths = glob(args['data'] + '/red_y_test/*.jpg')

elif not args["prototyping"]:
    x_train_paths = glob(args['data'] + '/x_train/*.jpg')
    y_train_paths = glob(args['data'] + '/y_train/*.jpg')
    x_val_paths = glob(args['data'] + '/x_val/*.jpg')
    y_val_paths = glob(args['data'] + '/y_val/*.jpg')
    x_test_paths = glob(args['data'] + '/x_test/*.jpg')
    y_test_paths = glob(args['data'] + '/y_test/*.jpg')

# Create Dataset using custom class in utils.dataset.py
transformations = transforms.ToTensor()

train = WaveSpectra(x_train_paths, y_train_paths, transform=transformations)
val = WaveSpectra(x_val_paths, y_val_paths, transform=transformations)
test = WaveSpectra(x_test_paths, y_test_paths, transform=transformations)

# Define DataLoaders for train/val/test sets
train_loader = DataLoader(train, batch_size=args['batch_size'], shuffle=True)
val_loader = DataLoader(val, batch_size=args['batch_size'], shuffle=True)
test_loader = DataLoader(test, batch_size=1, shuffle=False)

train_loader.requires_grad = True
val_loader.requires_grad = False
test_loader.requires_grad = False

# Save example image(s) from train set
x_train_sample, y_train_sample, x_train_index, y_train_index = next(iter(train_loader))
x_val_sample, y_val_sample, x_val_index, y_val_index = next(iter(val_loader))
x_test_sample, y_test_sample, x_test_index, y_test_index = next(iter(test_loader))

num_sample_images = 3 if args['batch_size'] >= 3 else args['batch_size']
for i in range(num_sample_images):
    plt.subplot(121, title='Offshore', xticks=[], yticks=[])
    plt.imshow(x_train_sample[i].squeeze(), cmap="gray")
    plt.subplot(122, title='Near Shore', xticks=[], yticks=[])
    plt.imshow(y_train_sample[i].squeeze(), cmap="gray")
    plt.suptitle("Example training data (normalised)")
    plt.tight_layout()
    plt.savefig(str(filepath) + "/sample_train" + str(i) + ".jpg")
    plt.close()

log.info("==========================================================================================")
# Display basic dataset information
if args['verbose']:
    # Precomputed values: RECALCULATE FOR YOUR DATASET WHEN PREPROCESSING
    mean = 0.1176469
    std = 0.00040002

    # Tabulate dataset info
    xtr1 = x_train_sample[0].squeeze()
    ytr1 = y_train_sample[0].squeeze()
    xva1 = x_val_sample[0].squeeze()
    yva1 = y_val_sample[0].squeeze()
    xte1 = x_test_sample[0].squeeze()
    yte1 = y_test_sample[0].squeeze()

    t1 = PrettyTable([' ', '# pairs', 'Dimensions', "# pixels", "Datatype"])
    t1.add_rows(
        [
            ["Train", str(len(train)), str(list(xtr1.size())), str(getinfo(xtr1)[1]), str(getinfo(xtr1)[2])],
            ["Validation", str(len(val)), str(list(xva1.size())), str(getinfo(xva1)[1]), str(getinfo(xva1)[2])],
            ["Test", str(len(test)), str(list(xte1.size())), str(getinfo(xte1)[1]), str(getinfo(xte1)[2])]
        ]
    )

    log.info("Basic dataset information:")
    log.info("Train set mean/std:\t" + str(mean) + "/" + str(std))
    log.info(t1)

log.info("==========================================================================================")
# Setup CNN using custom CAE class in models.model.py
network = WaveNet()
network.to(device)
writer = SummaryWriter(str(str(filepath) + "/logs/"))

if args['verbose']:
    log.debug(summary(network, (args['batch_size'], 1, 64, 64), verbose=2))

for param in network.parameters():
    param.requires_grad = True

total_samples = len(train)
n_iter = math.ceil(total_samples / args['batch_size'])

if args['verbose']:
    log.info("Number of epochs: " + str(args['num_epochs']))
    log.info("Total samples: " + str(total_samples))
    log.info("Batch size: " + str(args['batch_size']))
    log.info("# iterations: " + str(n_iter))

log.info("==========================================================================================")
# Set up training based on user-specified parameters
if str(args['optimizer']) == 'adam':
    optimizer = optim.Adam(network.parameters(),
                           lr=float(args['lr']),
                           betas=(args['momentum'], 0.999), )
elif str(args['optimizer']) == 'adamw':
    optimizer = optim.AdamW(network.parameters(),
                            lr=float(args['lr']),
                            betas=(args['momentum'], 0.999),
                            weight_decay=args['decay'])
elif str(args['optimizer']) == 'SGD':
    optimizer = optim.SGD(network.parameters(),
                          lr=float(args['lr']),
                          momentum=args['momentum'],
                          weight_decay=args['decay'])

if args['scheduler']:
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args['num_epochs'], eta_min=args['lr_min'])
    if args['verbose']:
        log.info("Using cosine learning rate annealing:")
        log.info("Max lr: {} - Min lr: {}".format(float(args['lr']), float(args['lr_min'])))

if args['loss'] == 'l1':
    loss_function = torch.nn.L1Loss(reduction=str(args['reduction']))
    torch.nn.L1Loss()
if args['loss'] == 'mse':
    loss_function = torch.nn.MSELoss(reduction=(str(args['reduction'])))
if args['loss'] == 'huber':
    loss_function = torch.nn.HuberLoss(reduction=str(args['reduction']))

test_interval = args['interval']

log.info('Setup complete!')
log.info("==========================================================================================")
log.info("==========================================================================================")

cos_losses, cos_sims, ssim_losses, ssim_sims = [], [], [], []
ssimLoss, ssimSim, ssim_val_losses, ssim_val_sim = [], [], [], []
cosineLoss, cosineSim, cos_val_losses, cos_val_sim = [], [], [], []
train_losses, val_losses, train_losses_iter, val_losses_iter = [], [], [], []
image_id, dimensions, l1_sum, l1_mean, l2_sum, l2_mean, huber_sum, huber_mean = \
    [], [], [], [], [], [], [], [],

def train():
    ssim = SSIM().cuda() if torch.cuda.is_available() else SSIM()
    last_loss = 0
    best_loss = 0
    log.info("Training Model...")
    
    for epoch in range(args['num_epochs']):
        i = 0
        train_losses_tmp = []
        val_losses_tmp = []
        cosine_losses_tmp = []
        cosine_similarity_tmp = []
        ssim_losses_tmp = []
        ssim_similarity_tmp = []
        cosine_similarity_val_tmp, cosine_losses_val_tmp = [], []
        ssim_similarity_val_tmp, ssim_losses_val_tmp = [], []
        
        with tqdm(train_loader, unit="batch") as tepoch:
            for data, target, data_index, target_index in tepoch:
                i += 1
                data = data.to(device)
                target = target.to(device)
                epoch_counter = f"Epoch {epoch + 1}/{args['num_epochs']}"
                tepoch.set_description(epoch_counter)
                optimizer.zero_grad()  # Clear gradients
                output = network(data)  # Forward pass
                
                # Compute cosine similarity
                target_vec = torch.flatten(target)
                output_vec = torch.flatten(output)

                cosine_sim = F.cosine_similarity(output_vec, target_vec, dim=0)  # Calculate loss
                cosine_loss = 1 - cosine_sim

                # Compute SSIM
                ssim_similarity = ssim(target, output)
                ssim_loss_initial = 1 - ssim_similarity

                if args['loss'] == 'cosine':
                    loss = cosine_loss
                elif args['loss'] == 'ssim':
                    loss = ssim_loss_initial
                else:
                    loss = loss_function(output, target)  # Calculate loss

                loss.backward()  # Calculate gradients
                
                train_losses_tmp.append(torch.Tensor.tolist(loss))
                cosine_similarity_tmp.append(cosine_sim.item() * 100)
                cosine_losses_tmp.append(cosine_loss.item())
                ssim_similarity_tmp.append(ssim_similarity * 100)
                ssim_losses_tmp.append(ssim_loss_initial.item())
                train_losses_iter.append(torch.Tensor.tolist(loss))
                
                optimizer.step()  # Update weights
                
                if args['verbose']:
                    tepoch.set_postfix(loss=loss.item())
                if str(args['track']) == data_index[0].split('.')[0]:
                    save_sample(data.detach().cpu().numpy(), target.detach().cpu().numpy(), output.detach().cpu().numpy().squeeze(), epoch, args['track'], filepath)
        
        train_loss = get_mean(train_losses_tmp[0:n_iter])
        train_losses.append(train_loss)
        cos_loss = get_mean(cosine_losses_tmp[0:n_iter])
        cos_losses.append(cosine_loss)
        cos_sim = get_mean(cosine_similarity_tmp[0:n_iter])
        cos_sims.append(cos_sim)
        ssim_loss = get_mean(ssim_losses_tmp[0:n_iter])
        ssim_losses.append(ssim_loss)
        ssim_sim = get_mean(ssim_similarity_tmp[0:n_iter])
        ssim_sims.append(ssim_sim)
        lr_schedule = optimizer.param_groups[0]["lr"]
        writer.add_scalar("Loss/train", loss, epoch)
        writer.add_scalar("Loss/cosine_error", cosine_loss.item(), epoch)
        writer.add_scalar("Loss/cosine_similarity", cosine_sim.item(), epoch)
        writer.add_scalar("Learning rate/Cosine", lr_schedule, epoch)
        if args['scheduler']:
            scheduler.step()

        prediction = output.detach().cpu().numpy().squeeze()
        train_path = (str(filepath) + "/predictions/" + "training/" + 'epoch_{}'.format(epoch + 1)).replace('\\', '/')
        val_path = (str(filepath) + "/predictions/" + "validation/" + 'epoch_{}'.format(epoch + 1)).replace('\\', '/')
        if not os.path.exists(train_path):
            os.mkdir(train_path)
        if not os.path.exists(val_path):
            os.mkdir(val_path)

        if len(prediction.shape) == 2:
            prediction_fixed = np.expand_dims(prediction, axis=0)
            prediction = prediction_fixed
        save_examples(data.detach().cpu().numpy().squeeze(), target.detach().cpu().numpy().squeeze(), prediction, 'Training', epoch, data_index, filepath)


        with torch.no_grad():
            with tqdm(val_loader, unit="batch") as tepoch:
                for data, target, data_index, target_index in tepoch:
                    data = data.to(device)
                    target = target.to(device)
                    indent = ' ' * len(epoch_counter)
                    tepoch.set_description(str(indent))
                    optimizer.zero_grad()
                    output = network(data)

                    # Flatten images -> 1D tensors: Compute cosine similarity/loss
                    target_vec = torch.flatten(target)
                    output_vec = torch.flatten(output)
                    cosine_sim = F.cosine_similarity(output_vec, target_vec, dim=0)  # Calculate loss
                    cosine_loss = 1 - cosine_sim

                    # Compute SSIM
                    ssim = SSIM().cuda() if torch.cuda.is_available() else SSIM()
                    ssim_similarity = ssim(target, output)
                    ssim_loss_initial = 1 - ssim_similarity

                    if args['loss'] == 'cosine':
                        loss = cosine_loss.item()
                    elif args['loss'] == 'ssim':
                        loss = ssim_loss_initial
                    else:
                        loss = loss_function(output, target)  # Calculate loss
                        
                    val_losses_tmp.append(torch.Tensor.tolist(loss))
                    val_losses_iter.append(torch.Tensor.tolist(loss))
                    cosine_similarity_val_tmp.append(cosine_sim.item() * 100)
                    cosine_losses_val_tmp.append(cosine_loss.item())
                    ssim_similarity_val_tmp.append(ssim_similarity * 100)
                    ssim_losses_val_tmp.append(ssim_loss_initial.item())
                    
                    if args['verbose']:
                        if args['loss'] == 'cosine':
                            tepoch.set_postfix(loss=loss)
                        else:
                            tepoch.set_postfix(loss=loss.item())
                            
                    if (epoch + 1) % (args['num_epochs']) == 0:
                        image_id.append(str(data_index[0]))
                        l1_sum.append(torch.Tensor.tolist(F.l1_loss(output, target, reduction='sum')))
                        l1_mean.append(torch.Tensor.tolist(F.l1_loss(output, target, reduction='mean')))
                        l2_sum.append(torch.Tensor.tolist(F.mse_loss(output, target, reduction='sum')))
                        l2_mean.append(torch.Tensor.tolist(F.mse_loss(output, target, reduction='mean')))
                        huber_sum.append(torch.Tensor.tolist(F.huber_loss(output, target, reduction='sum')))
                        huber_mean.append(torch.Tensor.tolist(F.huber_loss(output, target, reduction='mean')))
                        
                        cosineSim.append(cosine_sim)
                        cosineLoss.append(cosine_loss)

                        ssimLoss.append(ssim_loss_initial)
                        ssimSim.append(ssim_similarity)
                        
            prediction = output.detach().cpu().numpy().squeeze()
            if len(prediction.shape) == 2:
                prediction_fixed = np.expand_dims(prediction, axis=0)
                prediction = prediction_fixed
            save_examples(data.detach().cpu().numpy(),
                          target.detach().cpu().numpy(),
                          prediction,
                          'Validation',
                          epoch,
                          data_index,
                          filepath)
            writer.add_scalar("Loss/validation", loss / args['batch_size'], epoch)
            valid_loss = get_mean(val_losses_tmp[0:n_iter]) / args['batch_size']
            val_losses.append(valid_loss)
            writer.add_scalar("Loss/cosine_error", cosine_loss, epoch)
            writer.add_scalar("Loss/cosine_similarity", cosine_sim, epoch)
            writer.add_scalar("Loss/ssim_error", ssim_loss, epoch)
            writer.add_scalar("Loss/ssim_similarity", ssim_similarity, epoch)
            cos_sim_val = get_mean(cosine_similarity_val_tmp[0:n_iter]) / args['batch_size']
            cos_err_val = get_mean(cosine_losses_val_tmp[0:n_iter]) / args['batch_size']
            ssim_sim_val = get_mean(ssim_similarity_val_tmp[0:n_iter]) / args['batch_size']
            ssim_err_val = get_mean(ssim_losses_val_tmp[0:n_iter]) / args['batch_size']
            cos_val_losses.append(cos_err_val)
            cos_val_sim.append(cos_sim_val)
            ssim_val_losses.append(ssim_err_val)
            ssim_val_sim.append(ssim_sim_val)
            
            writer.add_scalars('Training Losses', {'Train_loss': train_loss,
                                                   'Valid_loss': valid_loss},
                               global_step=epoch)
            writer.add_scalars('Similarity (%)', {'SSIM (%)': ssim_sim_val,
                                                  'Cosine (%)': cos_sim_val},
                               global_step=epoch)
            writer.add_scalars('Error (%)', {'SSIM (%)': ssim_err_val,
                                             'Cosine (%)': cos_err_val},
                               global_step=epoch)

        if (epoch + 1) % test_interval == 0:
            torch.save(network.state_dict(), str(model_path) + "/last_model.pth")
            torch.save(optimizer.state_dict(), str(model_path) + "/last_optimizer.pth")
            if args['verbose']:
                log.info('Saved model checkpoint')

        if valid_loss < best_loss or epoch == 0:
            torch.save(network.state_dict(), str(model_path) + "/best_model.pth")
            torch.save(optimizer.state_dict(), str(model_path) + "/best_optimizer.pth")
            if args['verbose']:
                log.info('New best model and optimizer saved!')
            best_loss = valid_loss

        if args['verbose']:
            epoch_results = PrettyTable(['Learning Rate',
                                         'Train Loss',
                                         'Val Loss',
                                         "Cosine Loss (%)",
                                         "Cosine Similarity (%)",
                                         "SSIM Loss (%)",
                                         "SSIM Similarity (%)"])
            epoch_results.add_rows([['{:.4e}'.format(lr_schedule),
                                     '{:.6f}'.format(train_loss),
                                     '{:.6f}'.format(valid_loss),
                                     '{:.4f}'.format(cos_err_val * 100),
                                     '{:.4f}'.format(cos_sim_val),
                                     '{:.4f}'.format(ssim_err_val * 100),
                                     '{:.4f}'.format(ssim_sim_val)]])
        else:
            epoch_results = PrettyTable(['Train Loss',
                                         'Val Loss'])
            epoch_results.add_rows([['{:.6f}'.format(train_loss),
                                     '{:.6f}'.format(valid_loss)]])

        epoch_results.border = False
        log.info(epoch_results)

        if (epoch + 1) % (args['num_epochs']) == 0:
            log_dict = {
                "Epoch": (epoch + 1),
                "L1/MAE (sum)": get_mean(l1_sum),
                "L1/MAE (mean)": get_mean(l1_mean),
                "L2/MSE (sum)": get_mean(l2_sum),
                "L2/MSE (mean)": get_mean(l2_mean),
                "Huber (sum)": get_mean(huber_sum),
                "Huber (mean)": get_mean(huber_mean),
                "Cosine error": get_mean(cosineLoss).item(),
                "Cosine similarity": get_mean(cosineSim).item(),
                "SSIM error": get_mean(ssimLoss).item(),
                "SSIM similarity": get_mean(ssimSim).item()
            }
            df2 = DataFrame.from_dict([log_dict])
            if args['verbose']:
                log.info(df2)

    log.info("\n\nTraining complete!")
    print("\nBest results:")
    best_results = PrettyTable(['Train Loss',
                                 'Val Loss',
                                 "Cosine Loss (%)",
                                 "Cosine Similarity (%)",
                                 "SSIM Loss (%)",
                                 "SSIM Similarity (%)"])
    best_results.add_rows([['{:.6f}'.format(train_loss),
                            '{:.6f}'.format(valid_loss),
                            '{:.4f}'.format(cos_err_val * 100),
                            '{:.4f}'.format(cos_sim_val),
                            '{:.4f}'.format(ssim_err_val * 100),
                            '{:.4f}'.format(ssim_sim_val)]])
    log.info("Best training loss:\t{:.6f}\t"
             "Best validation loss:\t{:.6f}\t"
             "Best cosine loss:\t{:.6f}\t"
             "Best cosine similarity:\t{:.6f}"
             "Best SSIM loss:\t{:.6f}\t"
             "Best SSIM similarity:\t{:.6f}\n".format(min(train_losses),
                                                      min(val_losses),
                                                      min(cos_val_losses),
                                                      min(cos_val_sim),
                                                      min(ssim_val_losses),
                                                      min(ssim_val_sim)))
    log.info("==========================================================================================\n")

    return None


def evaluate():
    image_id.clear()
    l1_sum.clear()
    l1_mean.clear()
    l2_sum.clear()
    l2_mean.clear()
    huber_sum.clear()
    huber_mean.clear()
    cosineLoss.clear()
    cosineSim.clear()
    ssimLoss.clear()
    ssimSim.clear()
    log.info("Evaluating model...")

    model = WaveNet()
    if args['model_path'] is None:
        model.load_state_dict(torch.load(str(model_path) + "/best_model.pth"))
        if args['verbose']:
            log.info("model_path not supplied - loading : {}/best_model.pth".format(str(model_path).replace('\\', '/')))
    else:
        model.load_state_dict(torch.load(str(args['model_path'])))
        if args['verbose']:
            log.info("Loading model from supplied model_path: {}".format(str(args['model_path']).replace("\\", '/')))

    model.to(device)
    model.eval()

#     test_losses_l1, test_losses_mse, test_losses_huber = [], [], []
    cos_test_err, cos_test_sim, ssim_test_err, ssim_test_sim = [], [], [], []

    with tqdm(test_loader, unit="batch") as tepoch:
        with torch.no_grad():
            ssim = SSIM().cuda() if torch.cuda.is_available() else SSIM()
            i = 0
            for data, target, data_index, target_index in tepoch:
                data = data.to(device)
                target = target.to(device)
                tepoch.set_description(f"Evaluating")
                output = model(data)

                test_l1 = F.l1_loss(output, target, reduction='sum')
                test_mse = F.mse_loss(output, target, reduction='sum')
                test_huber = F.huber_loss(output, target, reduction='sum')
                
                # Flatten images -> 1D tensors: Compute cosine similarity/loss
                target_vec = torch.flatten(target)
                output_vec = torch.flatten(output)
                cosine_sim = F.cosine_similarity(output_vec, target_vec, dim=0)  # Calculate loss
                cosine_loss = 1 - cosine_sim

                # Compute SSIM
                ssim_similarity = ssim(target, output)
                ssim_loss_initial = 1 - ssim_similarity

                if args['loss'] == 'cosine':
                    loss = cosine_loss.item()
                elif args['loss'] == 'ssim':
                    loss = ssim_loss_initial
                else:
                    loss = loss_function(output, target)  # Calculate loss

                cosine_similarity_val_tmp.append(cosine_sim.item() * 100)
                cosine_losses_val_tmp.append(cosine_loss.item())
                ssim_similarity_val_tmp.append(ssim_similarity * 100)
                ssim_losses_val_tmp.append(ssim_loss_initial.item())

                prediction = output.detach().cpu().numpy().squeeze()
                save_inference(data.detach().cpu().numpy(),
                               target.detach().cpu().numpy(),
                               prediction,
                               torch.Tensor.tolist(test_l1),
                               torch.Tensor.tolist(test_mse),
                               torch.Tensor.tolist(test_huber),
                               data_index,
                               filepath)
                i += 1

                image_id.append(str(data_index[0]))
                l1_sum.append(torch.Tensor.tolist(F.l1_loss(output, target, reduction='sum')))
                l1_mean.append(torch.Tensor.tolist(F.l1_loss(output, target, reduction='mean')))
                l2_sum.append(torch.Tensor.tolist(F.mse_loss(output, target, reduction='sum')))
                l2_mean.append(torch.Tensor.tolist(F.mse_loss(output, target, reduction='mean')))
                huber_sum.append(torch.Tensor.tolist(F.huber_loss(output, target, reduction='sum')))
                huber_mean.append(torch.Tensor.tolist(F.huber_loss(output, target, reduction='mean')))
                
                cosineSim.append(cosine_sim)
                cosineLoss.append(cosine_loss)

                ssimLoss.append(ssim_loss_initial)
                ssimSim.append(ssim_similarity)

#                 test_losses_l1.append(torch.Tensor.tolist(test_l1))
#                 test_losses_mse.append(torch.Tensor.tolist(test_mse))
#                 test_losses_huber.append(torch.Tensor.tolist(test_huber))

    return None


def losses_to_csv(name):
    loss_dict = {
        "Image ID": image_id,
        "L1/MAE (sum)": l1_sum,
        "L1/MAE (mean)": l1_mean,
        "L2/MSE (sum)": l2_sum,
        "L2/MSE (mean)": l2_mean,
        "Huber (sum)": huber_sum,
        "Huber (mean)": huber_mean,
        "Cosine error": cosineLoss,
        "Cosine similarity": cosineSim,
        "SSIM error": ssimLoss,
        "SSIM similarity": ssimSim
    }
    df = DataFrame.from_dict(loss_dict)
    df = df.sort_values(by='L1/MAE (sum)', ascending=True)
    log.info(name + ":")
    log.info(df)

    df.to_csv(str(metrics_path) + "/" + name + ".csv", sep=",", float_format="%.6f", header=True, index=False)


# Execute program
train()
losses_to_csv('results_validation')
if args['num_epochs'] > 1:
    plot_losses(metrics_path, train_losses, val_losses)
evaluate()
losses_to_csv('results_evaluation')

if args['track']:
  pathname = preds_tr_path + "/" + args['track'] + "/"
  make_gif(pathname)
