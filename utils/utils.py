"""Convenience functions."""
import matplotlib.pyplot as plt
import torch


def sigmoid(x):
    return 1 / (1 + torch.exp(-8 * x + 4.5))


def get_mean(losses):
    return sum(losses) / len(losses)


def getinfo(img):
    (h, w) = img.shape
    total_px = img.shape[0] * img.shape[1]
    dtype = img.dtype
    return img.shape, total_px, dtype


def save_examples(x_data, y_data, z_data, mode, epoch, filename, root_path):
    for i in range(len(filename)):
        plt.subplot(131, title='Source', xticks=[], yticks=[])
        plt.imshow(x_data[i, :, :].squeeze(), cmap="gray")
        plt.subplot(132, title='Target', xticks=[], yticks=[])
        plt.imshow(y_data[i, :, :].squeeze(), cmap="gray")
        plt.subplot(133, title='Prediction', xticks=[], yticks=[])
        plt.imshow(z_data[i, :, :].squeeze(), cmap="gray")
        plt.suptitle(mode + " output: Epoch " + str(epoch + 1))
        plt.tight_layout()
        if mode == 'Training':
            plt.savefig(str(root_path) + "/predictions/training/epoch_" + str(epoch+1) + "/" + filename[i])
        elif mode == 'Validation':
            plt.savefig(str(root_path) + "/predictions/validation/epoch_" + str(epoch+1) + "/" + filename[i])
        plt.close()
    return None


def save_inference(x_data, y_data, z_data, loss1, loss2, loss3, filename, root_path):
    plt.subplot(131, title='Source', xticks=[], yticks=[])
    plt.imshow(x_data[0, :, :].squeeze(), cmap="gray")
    plt.subplot(132, title='Target', xticks=[], yticks=[])
    plt.imshow(y_data[0, :, :].squeeze(), cmap="gray")
    plt.subplot(133, title='Prediction', xticks=[], yticks=[])
    plt.imshow(z_data[:, :], cmap="gray")
    plt.suptitle("Evaluation:\nL1(MAE)/"
                 + str(round(loss1, 4))
                 + "\nL2(MSE)/" + str(round(loss2, 4))
                 + "\nHuber/" + str(round(loss3, 4)))
    plt.tight_layout()
    plt.savefig(str(root_path) + "/predictions/evaluation/" + str(filename[0]))
    plt.close()
    return None


def save_eval(x_data, y_data, z_data, loss1, loss2, loss3, filename, root_path):
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

    plt.subplot(131, title='Source', xticks=[], yticks=[])
    plt.imshow(x_data[:, :], cmap="gray")
    plt.subplot(132, title='Target', xticks=[], yticks=[])
    plt.imshow(y_data[:, :], cmap="gray")
    plt.subplot(133, title='Prediction', xticks=[], yticks=[])
    plt.imshow(z_data[:, :].squeeze(), cmap="gray")
    plt.suptitle("Evaluation:\nL1(MAE)/"
                 + str(round(loss1, 4))
                 + "\nL2(MSE)/" + str(round(loss2, 4))
                 + "\nHuber/" + str(round(loss3, 4)))
    plt.tight_layout()
    plt.savefig(str(root_path) + "/" + str(filename).split(".")[0] + "_comp.jpg")
    plt.close()
    return None


def save_error(target, prediction, error1, error2, error3, filename, root_path):
    plt.subplot(221, title='Target', xticks=[], yticks=[])
    plt.imshow(target[:, :], cmap="gray")
    plt.subplot(222, title='Prediction', xticks=[], yticks=[])
    plt.imshow(prediction[:, :], cmap="gray")

    plt.subplot(234, title='L1/MAE', xticks=[], yticks=[])
    plt.imshow(error1[:, :], cmap="jet")
    plt.subplot(235, title='L2/MSE', xticks=[], yticks=[])
    plt.imshow(error2[:, :], cmap="jet")
    plt.subplot(236, title='Huber', xticks=[], yticks=[])
    plt.imshow(error3[:, :], cmap="jet")

    plt.suptitle("Error maps for three losses:")
    plt.savefig(str(root_path) + "/" + str(filename).split(".")[0] + "_error.jpg")
    plt.close()

    return None


def plot_losses(metrics_path, train_losses, val_losses):
    xs = [x for x in range(len(train_losses))]
    plt.plot(xs, train_losses, color='blue', label='Train')
    plt.plot(xs, val_losses, color='red', label='Validation')
    plt.title("Train/Val losses")
    plt.xlabel("Epochs")
    plt.xticks(range(0, len(xs)))
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.legend()
    plt.savefig(str(metrics_path) + "/losses.jpg")

    return None


def save_prediction(prediction, filename, path):
    plt.imshow(prediction[:, :], cmap="gray")
    plt.axis('off')
    plt.suptitle("Error maps for three losses:")
    plt.savefig(str(path) + "/" + str(filename).split(".")[0] + ".jpg")
    plt.close()

    return None