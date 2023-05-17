import os, math, time, argparse, torch, torchvision, logging, pprint, warnings, numpy as np, pandas as pd
from torchmetrics import StructuralSimilarityIndexMeasure, CosineSimilarity
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from prettytable import PrettyTable
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchinfo import summary
from pandas import DataFrame
import torch.optim as optim
from pathlib import Path
from torch import Tensor
from tqdm import tqdm
from glob import glob
from PIL import Image
import torch.nn as nn


from utils.dataset import WaveSpectraNPZ, WaveSpectraNpzRAM, pad_img
from models.WaveNet import WaveNet
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
parser.add_argument("--track", action='store', type=str, required=False, default=None,
                    help="saves all predictions for a specified source image (file name only e.g. 00178)")
parser.add_argument("--cache", action="store_true", default=False,
                    help="load complete dataset into RAM")
parser.add_argument("--outputs", action="store_true", default=False,
                    help="save images from each train/validation epoch")
parser.add_argument("--num_workers", type=int, default=1,
                    help="number of workers")
parser.add_argument("--model_path", action="store", type=str, required=False,
                    help="path to saved model")
parser.add_argument("--resume", action="store_true", default=False,
                    help="path to saved model")
parser.add_argument("--device", type=str, default='cuda', choices=['cuda', 'cpu'],
                    help="device")
parser.add_argument("--optimizer", type=str, default='sgd', choices=['sgd', 'adamw'],
                    help="optimizer")
parser.add_argument("-m", "--momentum", type=float, default=0.9,
                    help="momentum for SGD, beta1 for adam")
parser.add_argument("--decay", type=float, default=0.0,
                    help="weight decay rate (default off)")
parser.add_argument("-e", "--num-epochs", type=int, default=20,
                    help="number of epochs")
parser.add_argument("-b", "--batch-size", type=int, default=1,
                    help="batch size")
parser.add_argument("--scheduler", action="store_false", default=True,
                    help="learning rate scheduler - cosine annealing (pass argument to disable)")
parser.add_argument("--lr", "--learning_rate", type=float, default=0.00001,
                    help="learning rate")
parser.add_argument("--lr_min", action="store", type=float, default=0.000005,
                    help="minimum learning rate for scheduler")
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

# Device configuration
device = torch.device('cuda' if args['device'] == 'cuda' and torch.cuda.is_available() else 'cpu')
if device == 'cuda':
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

x_train_path = args['data'] + "/x_train.npz"
y_train_path = args['data'] + "/y_train.npz"
x_val_path = args['data'] + "/x_val.npz"
y_val_path = args['data'] + "/y_val.npz"
x_test_path = args['data'] + "/x_test.npz"
y_test_path = args['data'] + "/y_test.npz"

# Create dataset
if args['cache']:
    train = WaveSpectraNpzRAM(x_train_path, y_train_path, device='cuda' if torch.cuda.is_available() else 'cpu')
    val = WaveSpectraNpzRAM(x_val_path, y_val_path, device='cuda' if torch.cuda.is_available() else 'cpu')
    test = WaveSpectraNpzRAM(x_test_path, y_test_path, device='cuda' if torch.cuda.is_available() else 'cpu')
else:
    with np.load(x_train_path) as x_train_data:
        x_train_data_files = x_train_data.files
    with np.load(y_train_path) as y_train_data:
        y_train_data_files = y_train_data.files
    with np.load(x_val_path) as x_val_data:
        x_val_data_files = x_val_data.files
    with np.load(y_val_path) as y_val_data:
        y_val_data_files = y_val_data.files
    with np.load(x_test_path) as x_test_data:
        x_test_data_files = x_test_data.files
    with np.load(y_test_path) as y_test_data:
        y_test_data_files = y_test_data.files

    train = WaveSpectraNPZ(x_train_path, y_train_path, x_train_data_files, y_train_data_files)
    val = WaveSpectraNPZ(x_val_path, y_val_path, x_val_data_files, y_val_data_files)
    test = WaveSpectraNPZ(x_test_path, y_test_path, x_test_data_files, y_test_data_files)

if args['resume']:
    checkpoint = torch.load(str(args['model_path']))

batch_size = args['batch_size'] if not args['resume'] else checkpoint['batch_size']

# Define DataLoaders
train_loader = DataLoader(train,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=args['num_workers'] if torch.cuda.is_available() else 0,
                          pin_memory=True if torch.cuda.is_available() else False)
val_loader = DataLoader(val,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=args['num_workers'] if torch.cuda.is_available() else 0,
                        pin_memory=True if torch.cuda.is_available() else False)
test_loader = DataLoader(test,
                         batch_size=1,
                         shuffle=False,
                         num_workers=args['num_workers'] if torch.cuda.is_available() else 0,
                         pin_memory=True if torch.cuda.is_available() else False)

train_loader.requires_grad = True
val_loader.requires_grad = False
test_loader.requires_grad = False


# Save example image(s) from train set
def dataset_info():
    x_train_sample, y_train_sample, sample_index, x_train_orig = next(iter(train_loader))
    x_val_sample, y_val_sample, sample_index, x_val_orig = next(iter(val_loader))
    x_test_sample, y_test_sample, sample_index, x_test_orig = next(iter(test_loader))

    num_sample_images = 3 if args['batch_size'] >= 3 else args['batch_size']
    for i in range(num_sample_images):
        plt.subplot(1, 3, 1)
        plt.title("Offshore")
        plt.xticks([])
        plt.yticks([])
        plt.imshow(x_train_orig[i].squeeze(), cmap="gray")

        plt.subplot(1, 3, (2, 3))
        plt.title("Near Shore")
        plt.xticks([])
        plt.yticks([])
        plt.imshow(y_train_sample[i].squeeze(), cmap="gray")

        plt.suptitle("Example training data (normalised)")
        plt.savefig(str(filepath) + "/sample_train" + str(i) + ".jpg")
        plt.close()

    # Display basic dataset information
    if args['verbose']:
        xtr1 = x_train_orig[0].squeeze()
        ytr1 = y_train_sample[0].squeeze()
        xva1 = x_val_sample[0].squeeze()
        yva1 = y_val_sample[0].squeeze()
        xte1 = x_test_sample[0].squeeze()
        yte1 = y_test_sample[0].squeeze()

        t1 = PrettyTable([' ', '# pairs', 'Dimensions (x)', 'Dimensions (y)', "# pixels", "Datatype"])
        t1.add_rows(
            [
                ["Train", str(len(train)), str(list(xtr1.size())), str(list(ytr1.size())), str(getinfo(xtr1)[1]), str(getinfo(xtr1)[2])],
                ["Validation", str(len(val)), str(list(xva1.size())), str(list(yva1.size())), str(getinfo(xva1)[1]), str(getinfo(xva1)[2])],
                ["Test", str(len(test)), str(list(xte1.size())), str(list(yte1.size())), str(getinfo(xte1)[1]), str(getinfo(xte1)[2])]
            ]
        )

        log.info("Basic dataset statistics:")
        log.info(t1)
        log.info("==========================================================================================")


# Setup CNN using custom CAE class in models.model.py
network = WaveNet()
network.to(device)
writer = SummaryWriter(str(str(filepath) + "/logs/"))

if args['verbose']:
    log.debug(summary(network, (args['batch_size'], 1, 29, 24), verbose=2))

for param in network.parameters():
    param.requires_grad = True

total_samples = len(train)
n_iter = math.ceil(total_samples // args['batch_size'])

optimizer = optim.AdamW(network.parameters(),
                        lr=float(args['lr']),
                        betas=(args['momentum'], 0.999),
                        weight_decay=args['decay'])

if args['optimizer'] == 'adamw':
    optimizer = optim.AdamW(network.parameters(),
                            lr=float(args['lr']),
                            betas=(args['momentum'], 0.999),
                            weight_decay=args['decay'])
elif args['optimizer'] == 'SGD':
    optimizer = optim.SGD(network.parameters(),
                          lr=float(args['lr']),
                          momentum=args['momentum'],
                          weight_decay=args['decay'])

if args['scheduler']:
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args['num_epochs'], eta_min=args['lr_min'])

test_interval = args['interval']

dataset_info()

if args['resume']:
    # load the model checkpoint
    log.info("Resuming training from saved checkpoint...")
    log.info("Loading checkpoint from supplied model_path: {}".format(str(args['model_path']).replace("\\", '/')))
    network.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    log.info("Model state_dict loaded")
    log.info("Optimizer state_dict loaded")
    log.info("Scheduler state_dict loaded")

    epochs = checkpoint['epoch']
    new_epochs = checkpoint['total_epochs'] - (epochs + 1)
    log.info(f"Previously trained for {epochs + 1}/{checkpoint['total_epochs']} epochs")
    log.info(f"Training for {new_epochs} more epochs...")

    batch_size = checkpoint['batch_size']

    log.info(f"\nCreated experiment directories at:\t{filepath}")
    log.info("\nDevice:\t" + str(device))
    if args['cache']:
        log.info("\nData stored in RAM for faster retrieval")
    else:
        log.info("\nLoading data lazily")

    if args['outputs']:
        log.info("\nSaving training/validation images each epoch")

    log.info("Total samples: \t\t" + str(total_samples))
    log.info("Batch size: \t\t" + str(batch_size))
    log.info("# iterations: \t\t" + str(n_iter))
    log.info('\nSetup complete!\n')

if args['verbose'] and not args['resume']:
    log.info("\nArguments parsed - Current configuration:")
    for k, v in args.items():
        log.info("{:<15} {:<10}".format(str(k), str(v)))

    log.info(f"\nCreated experiment directories at:\t{filepath}")
    log.info("\nDevice:\t" + str(device))
    if args['cache']:
        log.info("\nData stored in RAM for faster retrieval")
    else:
        log.info("\nLoading data lazily")

    if args['outputs']:
        log.info("\nSaving training/validation images each epoch")
    if args['scheduler']:
        log.info("\nUsing cosine learning rate annealing:")
        log.info("Max lr: {} - Min lr: {}".format(float(args['lr']), float(args['lr_min'])))
    log.info("\nOptimiser: {}".format(args['optimizer']))
    log.info("\nNumber of epochs: \t" + str(args['num_epochs']))
    log.info("Total samples: \t\t" + str(total_samples))
    log.info("Batch size: \t\t" + str(batch_size))
    log.info("# iterations: \t\t" + str(n_iter))
    log.info('\nSetup complete!\n')

log.info("==========================================================================================")

ssimLoss, ssimSim = [], []
cosineLoss, cosineSim = [], []
image_id, dimensions, l1_sum = [], [], []

training_losses, validation_losses = [], []
l1_training_losses, l1_validation_losses = [], []
cosine_training_losses, ssim_training_losses = [], []
cosine_validation_losses, ssim_validation_losses = [], []
cosine_training_similarities, ssim_training_similarities = [], []
cosine_validation_similarities, ssim_validation_similarities = [], []

ssim = StructuralSimilarityIndexMeasure(reduction='elementwise_mean').to(device)
cos = nn.CosineSimilarity(dim=1, eps=1e-6).to(device)
l1 = torch.nn.L1Loss(reduction='sum').to(device)

l1_start_epoch = 0


# # Clip model weights to [0, 1]
# class WeightClipper(object):
#     def __init__(self, frequency=5):
#         self.frequency = frequency
#
#     def __call__(self, module):
#         # filter the variables to get the ones you want
#         if hasattr(module, 'weight'):
#             w = module.weight.data
#             w = w.clamp(-1, 1)
#             module.weight.data = w
#
#
# clipper = WeightClipper()


def compute_cosine(output, target):
    target_vec = torch.flatten(target, start_dim=1, end_dim=3)
    output_vec = torch.flatten(output, start_dim=1, end_dim=3)
    cosine_similarity = cos(output_vec, target_vec)  # Calculate loss
    cosine_loss = 1 - cosine_similarity

    del target_vec
    del output_vec

    return cosine_similarity, cosine_loss


def compute_ssim(output, target):
    ssim_similarity = ssim(output, target)
    ssim_loss = 1 - ssim_similarity
    return ssim_similarity, ssim_loss


def train():
    last_loss = 0
    best_loss = 0
    no_improvement = 0
    log.info("Training Model...")
    check_l1 = True
    add_l1 = False

    if args['resume']:
        run_load = True

    if args['model_path'] and not args['resume']:
        network.load_state_dict(torch.load(str(args['model_path'])))
        if args['verbose']:
            log.info("Loading model from supplied model_path: {}".format(str(args['model_path']).replace("\\", '/')))

    for epoch in range(0 if not args['resume'] else (epochs + 1),
                       args['num_epochs'] if not args['resume'] else checkpoint['total_epochs']):
        if epoch > 1 and epoch % args['interval'] == 0:
            plot_losses(metrics_path, training_losses, validation_losses)
            plot_l1_losses(metrics_path, l1_training_losses, l1_validation_losses)
            plot_ssim_losses(metrics_path, ssim_training_losses, ssim_validation_losses)
            plot_cosine_losses(metrics_path, cosine_training_losses, cosine_validation_losses)
            plot_ssim_similarities(metrics_path, ssim_training_similarities, ssim_validation_similarities)
            plot_cosine_similarities(metrics_path, cosine_training_similarities, cosine_validation_similarities)

        if no_improvement == 5:
            print("Adding L1 loss...")
            add_l1 = True
            check_l1 = False
            global l1_start_epoch
            l1_start_epoch = epoch

        i = 0

        training_loss = 0
        l1_training_loss = 0
        ssim_training_loss = 0
        cosine_training_loss = 0
        ssim_training_similarity = 0
        cosine_training_similarity = 0

        validation_loss = 0
        l1_validation_loss = 0
        ssim_validation_loss = 0
        cosine_validation_loss = 0
        ssim_validation_similarity = 0
        cosine_validation_similarity = 0

        if args['resume'] and run_load:
            check_l1 = checkpoint['check_l1']
            add_l1 = checkpoint['add_l1']
            best_loss = checkpoint['best_loss']
            last_loss = checkpoint['last_loss']
            no_improvement = checkpoint['no_improvement']
            run_load = False

        with tqdm(train_loader, unit="batch") as tepoch:
            for data, target, index, data_orig in tepoch:
                i += 1
                data = data.to(device)
                target = target.to(device)
                if args['resume']:
                    epoch_counter = f"Epoch {epoch + 1}/{checkpoint['total_epochs']}"
                else:
                    epoch_counter = f"Epoch {epoch + 1}/{args['num_epochs']}"
                tepoch.set_description(epoch_counter)
                optimizer.zero_grad(set_to_none=True)  # Clear gradients
                output = network(data)  # Forward pass

                # Normalise output to same range as target
#                 target_min, target_max = target.min(), target.max()
#                 output = (output * (target_max - target_min)) + target_min

                # Compute cosine similarity:
                cosine_similarity, cosine_loss = compute_cosine(output, target)

                # Compute ssim
                ssim_similarity, ssim_loss = compute_ssim(output, target)

                # Compute l1 loss
                l1_loss = l1(output, target) / args['batch_size']

                if not add_l1:
                    loss = cosine_loss + ssim_loss
                    l1_loss.detach()
                else:
                    loss = cosine_loss + ssim_loss + (l1_loss * 0.1)

                training_loss += loss.detach().item()
                l1_training_loss += l1_loss.detach().item()
                ssim_training_loss += ssim_loss.detach().item()
                cosine_training_loss += cosine_loss.detach().item()
                ssim_training_similarity += ssim_similarity.detach().item()
                cosine_training_similarity += cosine_similarity.detach().item()

                loss.backward()  # Calculate gradients
                optimizer.step()  # Update weights

                if args['verbose']:
                    tepoch.set_postfix(loss=loss.item())

        if args['track']:
            x_track_orig = np.load(args['data'] + "/x_train.npz")[args['track']].astype(np.float32)
            y_track = np.load(args['data'] + "/y_train.npz")[args['track']].astype(np.float32)

            x_track_orig = torch.from_numpy(x_track_orig).to(device)
            x_padded = pad_img(x_track_orig).to(device)
            y_track = torch.from_numpy(y_track).to(device)
            y_track = torch.unsqueeze(y_track, dim=0)

            z_track = network(x_padded.unsqueeze(dim=0)).detach()
            save_sample(x_track_orig.detach().cpu().numpy(),
                        y_track.detach().cpu().numpy(),
                        z_track.cpu().numpy().squeeze(),
                        epoch,
                        args['track'],
                        filepath)

        training_loss /= n_iter
        l1_training_loss /= n_iter
        ssim_training_loss /= n_iter
        cosine_training_loss /= n_iter
        ssim_training_similarity /= n_iter
        cosine_training_similarity /= n_iter

        training_losses.append(training_loss)
        l1_training_losses.append(l1_training_loss)
        ssim_training_losses.append(ssim_training_loss)
        cosine_training_losses.append(cosine_training_loss)
        ssim_training_similarities.append(ssim_training_similarity)
        cosine_training_similarities.append(cosine_training_similarity)

        lr_schedule = optimizer.param_groups[0]["lr"]

        writer.add_scalar("Learning rate/Cosine", lr_schedule, epoch)

        if args['scheduler']:
            scheduler.step()

        prediction = output.detach().cpu().numpy().squeeze()

        train_path = (str(filepath) + "/predictions/" + "training/" + 'epoch_{}'.format(epoch + 1)).replace('\\', '/')
        val_path = (str(filepath) + "/predictions/" + "validation/" + 'epoch_{}'.format(epoch + 1)).replace('\\', '/')

        if len(prediction.shape) == 2:
            prediction_fixed = np.expand_dims(prediction, axis=0)
            prediction = prediction_fixed

        if args['outputs']:
            if not os.path.exists(train_path):
                os.mkdir(train_path)
            if not os.path.exists(val_path):
                os.mkdir(val_path)

            save_examples(data_orig.detach().cpu().numpy().squeeze(),
                          target.detach().cpu().numpy().squeeze(),
                          prediction,
                          'Training',
                          epoch,
                          index,
                          filepath,
                          args['batch_size'])

        # =============================================================================#
        iters = 0
        with torch.no_grad():
            with tqdm(val_loader, unit="batch") as tepoch:
                for data, target, index, data_orig in tepoch:
                    data = data.to(device)
                    target = target.to(device)
                    indent = ' ' * len(epoch_counter)
                    tepoch.set_description(str(indent))
                    optimizer.zero_grad()
                    output = network(data)

                    # Normalise output to same range as target
#                     target_min, target_max = target.min(), target.max()
#                     output = (output * (target_max - target_min)) + target_min

                    # Compute cosine similarity:
                    cosine_similarity, cosine_loss = compute_cosine(output, target)

                    # Compute ssim
                    ssim_similarity, ssim_loss = compute_ssim(output, target)

                    # Compute l1 loss
                    l1_loss = l1(output, target) / args['batch_size']

                    if not add_l1:
                        loss = cosine_loss + ssim_loss
                        l1_loss.detach()
                    else:
                        loss = cosine_loss + ssim_loss + (l1_loss * 0.1)

                    validation_loss += loss.item()
                    l1_validation_loss += l1_loss.item()
                    ssim_validation_loss += ssim_loss.item()
                    cosine_validation_loss += cosine_loss.item()
                    ssim_validation_similarity += ssim_similarity.item()
                    cosine_validation_similarity += cosine_similarity.item()

                    tepoch.set_postfix(loss=loss.item())

                    if (epoch + 1) % (args['num_epochs']) == 0:
                        image_id.append(str(index[0]))
                        l1_sum.append(l1_loss.item())
                        ssimLoss.append(torch.Tensor.tolist(ssim_loss))
                        cosineLoss.append(torch.Tensor.tolist(cosine_loss[0]))
                        ssimSim.append(torch.Tensor.tolist(ssim_similarity))
                        cosineSim.append(torch.Tensor.tolist(cosine_similarity[0]))

                    iters += 1  # for calculating means across validation iterations

            prediction = output.detach().cpu().numpy().squeeze()

            if len(prediction.shape) == 2:
                prediction_fixed = np.expand_dims(prediction, axis=0)
                prediction = prediction_fixed

            if args['outputs']:
                save_examples(data_orig.detach().cpu().numpy().squeeze(),
                              target.detach().cpu().numpy().squeeze(),
                              prediction,
                              'Validation',
                              epoch,
                              index,
                              filepath,
                              args['batch_size'])

            writer.add_scalar("Loss/validation", loss, epoch)

            validation_loss /= iters
            l1_validation_loss /= iters
            ssim_validation_loss /= iters
            cosine_validation_loss /= iters
            ssim_validation_similarity /= iters
            cosine_validation_similarity /= iters

            validation_losses.append(validation_loss)
            l1_validation_losses.append(l1_validation_loss)
            ssim_validation_losses.append(ssim_validation_loss)
            cosine_validation_losses.append(cosine_validation_loss)
            ssim_validation_similarities.append(ssim_validation_similarity)
            cosine_validation_similarities.append(cosine_validation_similarity)

            writer.add_scalar("Loss/l1_loss", l1_validation_loss, epoch)
            writer.add_scalar("Loss/ssim_error", ssim_validation_loss, epoch)
            writer.add_scalar("Loss/cosine_error", cosine_validation_loss, epoch)
            writer.add_scalar("Loss/ssim_similarity", ssim_validation_similarity, epoch)
            writer.add_scalar("Loss/cosine_similarity", cosine_validation_similarity, epoch)

            writer.add_scalars('Training Losses', {'Train_loss': training_loss,
                                                   'Valid_loss': validation_loss},
                               global_step=epoch)
            writer.add_scalars('SSIM Similarity', {'Training': ssim_training_similarity,
                                                   'Validation': ssim_validation_similarity},
                               global_step=epoch)
            writer.add_scalars('SSIM Loss', {'Training': ssim_training_loss,
                                             'Validation': ssim_validation_loss},
                               global_step=epoch)
            writer.add_scalars('Cosine Similarity', {'Training': cosine_training_similarity,
                                                     'Validation': cosine_validation_similarity},
                               global_step=epoch)
            writer.add_scalars('Cosine Loss', {'Training': cosine_training_loss,
                                               'Validation': cosine_validation_loss},
                               global_step=epoch)

        # save model checkpoint
        if (epoch + 1) % test_interval == 0:
            if epoch == 0:
                best_loss = last_loss = validation_loss

            total_epochs = checkpoint['total_epochs'] if args['resume'] else args['num_epochs']

            torch.save({'epoch': epoch,
                        'total_epochs': total_epochs,
                        'model_state_dict': network.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'check_l1': check_l1,
                        'add_l1': add_l1,
                        'best_loss': best_loss,
                        'last_loss': last_loss,
                        'no_improvement': no_improvement,
                        'batch_size': batch_size},
                       str(model_path) + "/last.pth")
            if args['verbose']:
                log.info('Saved model checkpoint')

        if no_improvement == 5:
            best_loss = validation_loss
            no_improvement = 0

        if validation_loss <= best_loss or epoch == 0:
            torch.save({'epoch': epoch,
                        'total_epochs': args['num_epochs'],
                        'model_state_dict': network.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'check_l1': check_l1,
                        'add_l1': add_l1,
                        'best_loss': validation_loss,
                        'last_loss': last_loss,
                        'no_improvement': no_improvement,
                        'batch_size': batch_size},
                       str(model_path) + "/best.pth")
            best_loss = validation_loss
            no_improvement = 0
            if args['verbose']:
                log.info('New best model and optimizer saved!')
        else:
            if check_l1:
                no_improvement += 1
                print(f"Epochs without improvement: {no_improvement}/5")

        epoch_results = PrettyTable(['Learning Rate',
                                     'Train Loss',
                                     'Val Loss',
                                     ' | ',
                                     "L1 Loss",
                                     " |",
                                     "Cosine Loss",
                                     "Cosine Similarity",
                                     '| ',
                                     "SSIM Loss",
                                     "SSIM Similarity"])
        epoch_results.add_rows([['{:.4e}'.format(lr_schedule),
                                 '{:.4f}'.format(training_loss),
                                 '{:.4f}'.format(validation_loss),
                                 ' | ',
                                 '{:.4f}'.format(l1_validation_loss),
                                 ' |',
                                 '{:.4f}'.format(cosine_validation_loss),
                                 '{:.4f}'.format(cosine_validation_similarity),
                                 '| ',
                                 '{:.4f}'.format(ssim_validation_loss),
                                 '{:.4f}'.format(ssim_validation_similarity)]])

        epoch_results.border = False
        log.info(epoch_results)
        log.info("\n")

    log.info("\n\nTraining complete!\n")

    log.info("Best training loss:\t{:.6f}\n"
             "Best validation loss:\t{:.6f}\n"
             "Best l1 loss:\t\t{:.6f}\n"
             "Best cosine loss:\t{:.6f}\n"
             "Best cosine similarity:\t{:.6f}\n"
             "Best SSIM loss:\t\t{:.6f}\n"
             "Best SSIM similarity:\t{:.6f}\n".format(min(training_losses),
                                                      min(validation_losses),
                                                      min(l1_validation_losses),
                                                      min(cosine_validation_losses),
                                                      max(cosine_validation_similarities),
                                                      min(ssim_validation_losses),
                                                      max(ssim_validation_similarities)))
    log.info("==========================================================================================\n")

    return None


def evaluate():
    image_id.clear()
    l1_sum.clear()
    cosineLoss.clear()
    cosineSim.clear()
    ssimLoss.clear()
    ssimSim.clear()
    log.info("Evaluating model...")

    model = WaveNet()
    if args['model_path'] is None:  # No model path specified: Use the current best model
        eval_checkpoint = torch.load(str(model_path))
        model.load_state_dict(eval_checkpoint['model_state_dict'])
        if args['verbose']:
            log.info("model_path not supplied - loading : {}/best.pth".format(str(model_path).replace('\\', '/')))
    elif args['model_path'] is not None and args['resume']:     # Model path specified, resume True: Use *non-argparse
        eval_checkpoint = torch.load(str(model_path) + "/best.pth")
        model.load_state_dict(checkpoint['model_state_dict'])
        log.info("Training resumed from previous - loading : {}/best.pth".format(str(model_path).replace('\\', '/')))
    else:   # Model path specified, resume False: Use arg['model_path']
        eval_checkpoint = torch.load(str(args['model_path']))
        model.load_state_dict(checkpoint['model_state_dict'])
        if args['verbose']:
            log.info("Loading model from supplied model_path: {}".format(str(args['model_path']).replace("\\", '/')))

    model.to(device)
    model.eval()

    with tqdm(test_loader, unit="sample") as tepoch:
        with torch.no_grad():
            i = 0
            for data, target, index, data_orig in tepoch:
                data = data.to(device)
                target = target.to(device)
                tepoch.set_description(f"Evaluating")
                output = model(data)

                test_l1 = l1(output, target)

                # Compute cosine similarity:
                cosine_similarity, cosine_loss = compute_cosine(output, target)

                # Compute ssim
                ssim_similarity, ssim_loss = compute_ssim(output, target)

                # Compute l1 loss
                l1_loss = l1(output, target)

                prediction = output.detach().cpu().numpy().squeeze()
                save_inference(data_orig.detach().cpu().numpy(),
                               target.detach().cpu().numpy(),
                               prediction,
                               torch.Tensor.tolist(test_l1),
                               torch.Tensor.tolist(ssim_similarity),
                               torch.Tensor.tolist(cosine_similarity[0]),
                               index,
                               filepath)
                i += 1

                image_id.append(str(index[0]))
                l1_sum.append(torch.Tensor.tolist(l1_loss))
                ssimLoss.append(torch.Tensor.tolist(ssim_loss))
                cosineLoss.append(torch.Tensor.tolist(cosine_loss[0]))
                ssimSim.append(torch.Tensor.tolist(ssim_similarity))
                cosineSim.append(torch.Tensor.tolist(cosine_similarity[0]))

    return None


def losses_to_csv(name):
    loss_dict = {
        "Image ID": image_id,
        "L1/MAE": l1_sum,
        "Cosine error": cosineLoss,
        "Cosine similarity": cosineSim,
        "SSIM error": ssimLoss,
        "SSIM similarity": ssimSim
    }
    df = DataFrame.from_dict(loss_dict)
    df = df.sort_values(by='L1/MAE', ascending=True)
    df = df.round(6)
    log.info(name + ":")
    log.info(df)

    df.to_csv(str(metrics_path) + "/" + name + ".csv", sep=",", float_format="%.6f", header=True, index=False)
    log.info("Saved as " + str(metrics_path) + "/" + name + ".csv")


# Execute program
train()
losses_to_csv('results_validation')

if args['num_epochs'] > 1:
    plot_final_losses(metrics_path, training_losses, validation_losses, l1_start_epoch)
    plot_l1_losses(metrics_path, l1_training_losses, l1_validation_losses)
    plot_ssim_losses(metrics_path, ssim_training_losses, ssim_validation_losses)
    plot_cosine_losses(metrics_path, cosine_training_losses, cosine_validation_losses)
    plot_ssim_similarities(metrics_path, ssim_training_similarities, ssim_validation_similarities)
    plot_cosine_similarities(metrics_path, cosine_training_similarities, cosine_validation_similarities)

evaluate()
losses_to_csv('results_evaluation')

if args['track']:
    pathname = str(preds_tr_path) + "/" + str(args['track']) + "/"
    make_gif(str(pathname), str(preds_tr_path), str(args['track']))
