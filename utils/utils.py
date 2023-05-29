"""Misc. functions"""
import matplotlib.pyplot as plt
from pathlib import Path
import torch.nn as nn
from PIL import Image
from glob import glob
import torch
import os


def sigmoid(x):
    return 1 / (1 + torch.exp(-8 * x + 4.5))


def getinfo(img):
    (h, w) = img.shape
    total_px = img.shape[0] * img.shape[1]
    dtype = img.dtype
    return img.shape, total_px, dtype


def init_layers(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight)
        # torch.nn.init.constant_(m.weight, 0.001)
        torch.nn.init.zeros_(m.bias)
    return None


def make_gif(frame_folder, savepath, foldername):
    pathname = savepath + "/" + foldername + "/"
    frames = [Image.open(image) for image in glob(f"{frame_folder}/*.jpg")]
    frame_one = frames[0]
    frame_one.save(pathname + "training.gif", format="gif", append_images=frames,
                   save_all=True, duration=200, loop=0)

    return None


def normalise_to_source(target, output):
    output_min, output_max = output.min(), output.max()
    target_min, target_max = target.min(), target.max()
    output = (output - output_min) / (output_max - output_min) * (target_max - target_min) + target_min

    return output


def min_max_01(img):
    img_min, img_max = img.min(), img.max()
    img = (img - img_min) / (img_max - img_min) * (1 - 0) + 0

    return img, img_min, img_max


def min_max_revert(img, og_min, og_max):
    og_img = (img * (og_max - og_min)) + og_min

    return og_img


def save_sample(data, target, prediction, epoch, filename, root_path):
    epoch_str = str(epoch + 1).zfill(3)

    plt.subplot(2, 3, (1, 4))
    plt.title("Source")
    plt.xticks([])
    plt.yticks([])
    plt.imshow(data[:, :].squeeze(), cmap="gray")

    plt.subplot(2, 3, (2, 3))
    plt.title("Target")
    plt.xticks([])
    plt.yticks([])
    plt.imshow(target[0, :, :].squeeze(), cmap="gray")

    plt.subplot(2, 3, (5, 6))
    plt.title("Prediction")
    plt.xticks([])
    plt.yticks([])
    plt.imshow(prediction[:, :].squeeze(), cmap="gray")

    plt.suptitle("Output: Epoch " + str(epoch + 1))
    plt.tight_layout()

    epoch_str = "epoch_" + epoch_str
    filename_path = str(filename)
    pathname = "/predictions/" + "training/" + filename_path + "/" + epoch_str
    fullpath = str(root_path) + pathname

    if not os.path.exists(os.path.dirname(fullpath)):
        os.mkdir(os.path.dirname(fullpath))

    plt.savefig(fullpath + ".jpg")
    plt.close()
    return None


def save_examples(x_data, y_data, z_data, mode, epoch, filename, root_path, batch):

    if batch > 1:
        for i in range(len(filename)):
            plt.subplot(2, 3, (1, 4))
            plt.title("Source")
            plt.xticks([])
            plt.yticks([])
            plt.imshow(x_data[i, :, :].squeeze(), cmap="gray")

            plt.subplot(2, 3, (2, 3))
            plt.title("Target")
            plt.xticks([])
            plt.yticks([])
            plt.imshow(y_data[i, :, :].squeeze(), cmap="gray")

            plt.subplot(2, 3, (5, 6))
            plt.title("Prediction")
            plt.xticks([])
            plt.yticks([])
            plt.imshow(z_data[i, :, :].squeeze(), cmap="gray")

            plt.suptitle(mode + " output: Epoch " + str(epoch + 1))
            # plt.tight_layout()
            if mode == 'Training':
                plt.savefig(str(root_path) + "/predictions/training/epoch_" + str(epoch+1) + "/" + filename[i])
            elif mode == 'Validation':
                plt.savefig(str(root_path) + "/predictions/validation/epoch_" + str(epoch+1) + "/" + filename[i])
            plt.close()
    else:
        plt.subplot(2, 3, (1, 4))
        plt.title("Source")
        plt.xticks([])
        plt.yticks([])
        plt.imshow(x_data[:, :], cmap="gray")

        plt.subplot(2, 3, (2, 3))
        plt.title("Target")
        plt.xticks([])
        plt.yticks([])
        plt.imshow(y_data[:, :], cmap="gray")

        plt.subplot(2, 3, (5, 6))
        plt.title("Prediction")
        plt.xticks([])
        plt.yticks([])
        plt.imshow(z_data[:, :, :].squeeze(), cmap="gray")

        plt.suptitle(mode + " output: Epoch " + str(epoch + 1))

        if mode == 'Training':
            plt.savefig(str(root_path) + "/predictions/training/epoch_" + str(epoch+1) + "/" + filename[0])
        elif mode == 'Validation':
            plt.savefig(str(root_path) + "/predictions/validation/epoch_" + str(epoch+1) + "/" + filename[0])
        plt.close()

    return None


def save_inference(x_data, y_data, z_data, loss1, ssim, cosine, filename, root_path):
    plt.subplot(2, 3, (1, 4))
    plt.title("Source")
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_data[0, :, :].squeeze(), cmap="gray")

    plt.subplot(2, 3, (2, 3))
    plt.title("Target")
    plt.xticks([])
    plt.yticks([])
    plt.imshow(y_data[0, :, :].squeeze(), cmap="gray")

    plt.subplot(2, 3, (5, 6))
    plt.title("Prediction")
    plt.xticks([])
    plt.yticks([])
    plt.imshow(z_data[:, :], cmap="gray")

    plt.suptitle("Evaluation:\nL1(MAE): "
                 + str(round(loss1, 4))
                 + "/SSIM: "
                 + str(round(ssim, 4))
                 + "/Cosine: "
                 + str(round(cosine, 4)))

    plt.savefig(str(root_path) + "/predictions/evaluation/" + str(filename[0]))
    plt.close()

    return None


def save_eval(x_data, y_data, z_data, loss1, ssim, cosine, filename, root_path):
    plt.imshow(x_data[:, :], cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(str(root_path) + "/" + str(filename).split(".")[0] + "_source.jpg")
    plt.close()

    plt.imshow(y_data[:, :], cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(str(root_path) + "/" + str(filename).split(".")[0] + "_target.jpg")
    plt.close()

    plt.imshow(z_data[:, :].squeeze(), cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(str(root_path) + "/" + str(filename).split(".")[0] + "_pred.jpg")
    plt.close()

    plt.subplot(2, 3, (1, 4))
    plt.title("Source")
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_data[:, :], cmap="gray")

    plt.subplot(2, 3, (2, 3))
    plt.title("Target")
    plt.xticks([])
    plt.yticks([])
    plt.imshow(y_data[:, :], cmap="gray")

    plt.subplot(2, 3, (5, 6))
    plt.title("Prediction")
    plt.xticks([])
    plt.yticks([])
    plt.imshow(z_data[:, :].squeeze(), cmap="gray")

    plt.suptitle("Evaluation:\nL1(MAE): "
                 + str(round(loss1, 4))
                 + "/SSIM: "
                 + str(round(ssim, 4))
                 + "/Cosine: "
                 + str(round(cosine, 4)))

    plt.savefig(str(root_path) + "/" + str(filename).split(".")[0] + "_comp.jpg")
    plt.close()
    return None


def plot_losses(metrics_path, train_losses, val_losses):
    xs = [x for x in range(len(train_losses))]
    plt.plot(xs, train_losses, color='blue', label='Train')
    plt.plot(xs, val_losses, color='red', label='Validation')
    plt.title("Train/Validation losses")
    plt.xlabel("Epochs")
    plt.xticks(range(0, len(xs)))
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.legend()
    plt.savefig(str(metrics_path) + "/metrics_losses.jpg")
    plt.close()
    return None


def plot_l1_losses(metrics_path, l1_training_losses, l1_validation_losses):
    xs = [x for x in range(len(l1_training_losses))]
    plt.plot(xs, l1_training_losses, color='blue', label='Train')
    plt.plot(xs, l1_validation_losses, color='red', label='Validation')
    plt.title("L1 loss")
    plt.xlabel("Epochs")
    plt.xticks(range(0, len(xs)))
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.legend()
    plt.savefig(str(metrics_path) + "/metrics_losses_l1.jpg")
    plt.close()
    return None


def plot_ssim_losses(metrics_path, ssim_training_losses, ssim_validation_losses):
    xs = [x for x in range(len(ssim_training_losses))]
    plt.plot(xs, ssim_training_losses, color='blue', label='Train')
    plt.plot(xs, ssim_validation_losses, color='red', label='Validation')
    plt.title("SSIM loss")
    plt.xlabel("Epochs")
    plt.xticks(range(0, len(xs)))
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.legend()
    plt.savefig(str(metrics_path) + "/metrics_losses_ssim.jpg")
    plt.close()
    return None


def plot_ssim_similarities(metrics_path, ssim_training_similarities, ssim_validation_similarities):
    xs = [x for x in range(len(ssim_training_similarities))]
    plt.plot(xs, ssim_training_similarities, color='blue', label='Train')
    plt.plot(xs, ssim_validation_similarities, color='red', label='Validation')
    plt.title("SSIM similarity")
    plt.xlabel("Epochs")
    plt.xticks(range(0, len(xs)))
    plt.ylabel("Similarity")
    plt.tight_layout()
    plt.legend()
    plt.savefig(str(metrics_path) + "/metrics_similarity_ssim.jpg")
    plt.close()
    return None


def plot_cosine_losses(metrics_path, cosine_training_losses, cosine_validation_losses):
    xs = [x for x in range(len(cosine_training_losses))]
    plt.plot(xs, cosine_training_losses, color='blue', label='Train')
    plt.plot(xs, cosine_validation_losses, color='red', label='Validation')
    plt.title("Cosine loss")
    plt.xlabel("Epochs")
    plt.xticks(range(0, len(xs)))
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.legend()
    plt.savefig(str(metrics_path) + "/metrics_losses_cosine.jpg")
    plt.close()
    return None


def plot_cosine_similarities(metrics_path, cosine_training_similarities, cosine_validation_similarities):
    xs = [x for x in range(len(cosine_training_similarities))]
    plt.plot(xs, cosine_training_similarities, color='blue', label='Train')
    plt.plot(xs, cosine_validation_similarities, color='red', label='Validation')
    plt.title("Cosine similarity")
    plt.xlabel("Epochs")
    plt.xticks(range(0, len(xs)))
    plt.ylabel("Similarity")
    plt.tight_layout()
    plt.legend()
    plt.savefig(str(metrics_path) + "/metrics_similarity_cosine.jpg")
    plt.close()
    return None


def plot_final_losses(metrics_path, train_losses, val_losses, epoch):
    xs = [x for x in range(len(train_losses))]
    plt.plot(xs, train_losses, color='blue', label='Train')
    plt.plot(xs, val_losses, color='red', label='Validation')
    plt.title("Train/Val losses")
    plt.xlabel("Epochs")
    plt.xticks(range(0, len(xs)))
    plt.ylabel("Loss")
    if epoch != 0:
        plt.axvline(x=epoch, color='purple', label='L1 loss added', ls=':')
    plt.tight_layout()
    plt.legend()
    plt.savefig(str(metrics_path) + "/metrics_losses.jpg")
    plt.close()
    return None


def save_prediction(prediction, filename, path):
    plt.imshow(prediction[:, :], cmap="gray")
    plt.axis('off')
    plt.savefig(str(path) + "/" + str(filename).split(".")[0] + ".jpg", bbox_inches="tight", pad_inches=0)
    plt.close()

    return None
