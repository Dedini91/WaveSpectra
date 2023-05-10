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
from torchmetrics import StructuralSimilarityIndexMeasure, CosineSimilarity, MeanAbsoluteError

from models.WaveNet import WaveNet
from utils.dataset import WaveSpectra, pad_img
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
parser.add_argument("-e", "--num-epochs", type=int, default=20,
                    help="number of epochs")
parser.add_argument("-b", "--batch-size", type=int, default=1,
                    help="batch size")
parser.add_argument("--lr", "--learning_rate", type=float, default=0.00002,
                    help="learning rate")
parser.add_argument("-m", "--momentum", type=float, default=0.9,
                    help="momentum for SGD, beta1 for adam")
parser.add_argument("--scheduler", action="store_false", default=True,
                    help="learning rate scheduler - cosine annealing (pass argument to disable)")
parser.add_argument("--lr_min", action="store", type=float, default=0.000005,
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
else:
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
x_train_sample, y_train_sample, x_train_index, y_train_index, x_train_orig = next(iter(train_loader))
x_val_sample, y_val_sample, x_val_index, y_val_index, x_val_orig = next(iter(val_loader))
x_test_sample, y_test_sample, x_test_index, y_test_index, x_test_orig = next(iter(test_loader))

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

log.info("==========================================================================================")
# Display basic dataset information
if args['verbose']:
    # Precomputed values: RECALCULATE FOR YOUR DATASET WHEN PREPROCESSING
    # mean = 0.1176469
    # std = 0.00040002

    # Tabulate dataset info
    xtr1 = x_train_orig[0].squeeze()
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
    # log.info("Train set mean/std:\t" + str(mean) + "/" + str(std))
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
n_iter = math.ceil(total_samples / args['batch_size'])

if args['verbose']:
    log.info("Number of epochs: " + str(args['num_epochs']))
    log.info("Total samples: " + str(total_samples))
    log.info("Batch size: " + str(args['batch_size']))
    log.info("# iterations: " + str(n_iter))

log.info("==========================================================================================")
optimizer = optim.AdamW(network.parameters(),
                        lr=float(args['lr']),
                        betas=(args['momentum'], 0.999),
                        weight_decay=args['decay'])

if args['scheduler']:
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args['num_epochs'], eta_min=args['lr_min'])
    if args['verbose']:
        log.info("Using cosine learning rate annealing:")
        log.info("Max lr: {} - Min lr: {}".format(float(args['lr']), float(args['lr_min'])))

test_interval = args['interval']

log.info('Setup complete!\n')
log.info("==========================================================================================")
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

ssim = StructuralSimilarityIndexMeasure(reduction='elementwise_mean')
cos = CosineSimilarity(reduction='mean')
mae = MeanAbsoluteError()
l1 = torch.nn.L1Loss(reduction='sum')

if args['track']:
    x_track_orig = Image.open('data/processed/x_train/' + str(args['track']) + '.jpg')
    y_track = Image.open('data/processed/y_train/' + str(args['track']) + '.jpg')

    x_track_orig = transformations(x_track_orig).to(device)
    x_padded = pad_img(x_track_orig).to(device)
    y_track = transformations(y_track).to(device)

l1_start_epoch = 0


def train():
    last_loss = 0
    best_loss = 0
    no_improvement = 0
    log.info("Training Model...")
    check_l1 = True
    add_l1 = False

    for epoch in range(args['num_epochs']):
        if epoch > 1 and epoch % 5 == 0:
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

        with tqdm(train_loader, unit="batch") as tepoch:
            for data, target, data_index, target_index, data_orig in tepoch:
                i += 1
                data = data.to(device)
                target = target.to(device)
                epoch_counter = f"Epoch {epoch + 1}/{args['num_epochs']}"
                tepoch.set_description(epoch_counter)
                optimizer.zero_grad()  # Clear gradients
                output = network(data)  # Forward pass

                # Compute cosine similarity:
                target_vec = torch.flatten(target, start_dim=1, end_dim=3)
                output_vec = torch.flatten(output, start_dim=1, end_dim=3)
                cosine_similarity = cos(output_vec, target_vec)  # Calculate loss
                cosine_loss = 1 - cosine_similarity

                # Compute SSIM
                ssim_similarity = ssim(output, target)
                ssim_loss = 1 - ssim_similarity

                # Compute l1 loss
                l1_loss = l1(output, target) / args['batch_size']

                if not add_l1:
                    loss = cosine_loss + ssim_loss
                else:
                    loss = cosine_loss + ssim_loss + (l1_loss * 0.01)

                loss.backward()  # Calculate gradients
                optimizer.step()  # Update weights

                training_loss += loss.item()
                l1_training_loss += l1_loss.item()
                ssim_training_loss += ssim_loss.item()
                cosine_training_loss += cosine_loss.item()
                ssim_training_similarity += ssim_similarity.item()
                cosine_training_similarity += cosine_similarity.item()

                if args['verbose']:
                    tepoch.set_postfix(loss=loss.item())

        # if str(args['track']) == data_index[0].split('.')[0]:
        if str(args['track']):
            z_track = network(x_padded.unsqueeze(dim=0))

            save_sample(x_track_orig.detach().cpu().numpy(),
                        y_track.detach().cpu().numpy(),
                        z_track.detach().cpu().numpy().squeeze(),
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

        if not os.path.exists(train_path):
            os.mkdir(train_path)
        if not os.path.exists(val_path):
            os.mkdir(val_path)

        if len(prediction.shape) == 2:
            prediction_fixed = np.expand_dims(prediction, axis=0)
            prediction = prediction_fixed

        save_examples(data_orig.detach().cpu().numpy().squeeze(),
                      target.detach().cpu().numpy().squeeze(),
                      prediction,
                      'Training',
                      epoch,
                      data_index,
                      filepath,
                      args['batch_size'])

        # =============================================================================#
        iters = 0
        with torch.no_grad():
            with tqdm(val_loader, unit="batch") as tepoch:
                for data, target, data_index, target_index, data_orig in tepoch:
                    data = data.to(device)
                    target = target.to(device)
                    indent = ' ' * len(epoch_counter)
                    tepoch.set_description(str(indent))
                    optimizer.zero_grad()
                    output = network(data)

                    # Flatten images -> 1D tensors: Compute cosine similarity/loss
                    target_vec = torch.flatten(target, start_dim=1, end_dim=3)
                    output_vec = torch.flatten(output, start_dim=1, end_dim=3)
                    cosine_similarity = cos(output_vec, target_vec)
                    cosine_loss = 1 - cosine_similarity

                    # Compute SSIM
                    ssim_similarity = ssim(target, output)
                    ssim_loss = 1 - ssim_similarity

                    # Compute l1 loss
                    l1_loss = l1(output, target) / args['batch_size']

                    if not add_l1:
                        loss = cosine_loss + ssim_loss
                    else:
                        loss = cosine_loss + ssim_loss + (l1_loss * 0.01)

                    validation_loss += loss.item()
                    l1_validation_loss += l1_loss.item()
                    ssim_validation_loss += ssim_loss.item()
                    cosine_validation_loss += cosine_loss.item()
                    ssim_validation_similarity += ssim_similarity.item()
                    cosine_validation_similarity += cosine_similarity.item()

                    tepoch.set_postfix(loss=loss.item())

                    if (epoch + 1) % (args['num_epochs']) == 0:
                        image_id.append(str(data_index[0]))
                        l1_sum.append(l1_loss.item())
                        cosineSim.append(cosine_similarity.detach().cpu().numpy())
                        cosineLoss.append(cosine_loss.detach().cpu().numpy())
                        ssimLoss.append(ssim_loss.detach().cpu().numpy())
                        ssimSim.append(ssim_similarity.detach().cpu().numpy())

                    iters += 1  # for calculating means across validation iterations

            prediction = output.detach().cpu().numpy().squeeze()

            if len(prediction.shape) == 2:
                prediction_fixed = np.expand_dims(prediction, axis=0)
                prediction = prediction_fixed

            save_examples(data_orig.detach().cpu().numpy().squeeze(),
                          target.detach().cpu().numpy().squeeze(),
                          prediction,
                          'Validation',
                          epoch,
                          data_index,
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
            writer.add_scalars('Similarity', {'SSIM': ssim_validation_similarity,
                                              'Cosine': cosine_validation_similarity},
                               global_step=epoch)
            writer.add_scalars('Error', {'SSIM': ssim_validation_loss,
                                         'Cosine': cosine_validation_loss},
                               global_step=epoch)

        if (epoch + 1) % test_interval == 0:
            torch.save(network.state_dict(), str(model_path) + "/last_model.pth")
            torch.save(optimizer.state_dict(), str(model_path) + "/last_optimizer.pth")
            if args['verbose']:
                log.info('Saved model checkpoint')

        if no_improvement == 5:
            best_loss = validation_loss
            no_improvement = 0

        if validation_loss <= best_loss or epoch == 0:
            torch.save(network.state_dict(), str(model_path) + "/best_model.pth")
            torch.save(optimizer.state_dict(), str(model_path) + "/best_optimizer.pth")
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
             "Best l1 loss:\t{:.6f}\n"
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

    with tqdm(test_loader, unit="batch") as tepoch:
        with torch.no_grad():
            i = 0
            for data, target, data_index, target_index, data_orig in tepoch:
                data = data.to(device)
                target = target.to(device)
                tepoch.set_description(f"Evaluating")
                output = model(data)

                test_l1 = l1(output, target)

                # Flatten images -> 1D tensors: Compute cosine similarity/loss
                target_vec = torch.flatten(target)
                output_vec = torch.flatten(output)
                cosine_sim = cos(output_vec, target_vec)  # Calculate loss
                cosine_loss = 1 - cosine_sim

                # Compute SSIM
                ssim_similarity = ssim(target, output)
                ssim_loss = 1 - ssim_similarity

                # Compute l1 loss
                l1_loss = l1(output, target)

                prediction = output.detach().cpu().numpy().squeeze()
                save_inference(data_orig.detach().cpu().numpy(),
                               target.detach().cpu().numpy(),
                               prediction,
                               torch.Tensor.tolist(test_l1),
                               torch.Tensor.tolist(ssim_similarity),
                               torch.Tensor.tolist(cosine_sim),
                               data_index,
                               filepath)
                i += 1

                image_id.append(str(data_index[0]))
                l1_sum.append(torch.Tensor.tolist(l1_loss))
                cosineSim.append(torch.Tensor.tolist(cosine_sim))
                cosineLoss.append(torch.Tensor.tolist(cosine_loss))
                ssimLoss.append(torch.Tensor.tolist(ssim_loss))
                ssimSim.append(torch.Tensor.tolist(ssim_similarity))

    return None


def losses_to_csv(name):
    loss_dict = {
        "Image ID": image_id,
        "L1 loss": l1_sum,
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
