import numpy as np
import os
import cv2 as cv
from pathlib import Path
from tqdm import tqdm
import random
import shutil
import argparse
from img2vec_pytorch import Img2Vec
from sklearn.cluster import KMeans
from itertools import compress
import matplotlib.pyplot as plt
from utils import clustering, clustering_utils as utils
import torch
import warnings
from k_means_constrained import KMeansConstrained

warnings.filterwarnings('ignore', category=UserWarning, message="The parameter 'pretrained' is deprecated since 0.13 "
                                                                "and may be removed in the future, please use 'weights'"
                                                                " instead.")
warnings.filterwarnings('ignore', category=UserWarning, message="Arguments other than a weight enum or `None` for "
                                                                "'weights' are deprecated since 0.13 and may be "
                                                                "removed in the future. The current behavior is "
                                                                "equivalent to passing `weights=ResNet18_Weights."
                                                                "IMAGENET1K_V1`. You can also use `weights=ResNet18_"
                                                                "Weights.DEFAULT` to get the most up-to-date weights.")
warnings.filterwarnings('ignore', category=FutureWarning, message="The default value of `n_init` will change from 10 "
                                                                  "to 'auto' in 1.4. Set the value of `n_init` "
                                                                  "explicitly to suppress the warning")


parser = argparse.ArgumentParser(description="Preprocessing options")

parser.add_argument("-d", "--data", action="store", type=str, required=True,
                    help="path to parent directory of raw data folder")
parser.add_argument("-n", "--name", action="store", type=str, required=True,
                    help="dataset name")
parser.add_argument("-s", type=int, default=2000, required=True,
                    help="number of training samples (max 80% of total # samples")
parser.add_argument("-c", type=int, default=10, required=True,
                    help="number of clusters (classes)")

args = vars(parser.parse_args())

root_path = Path(args['data'])

numSamples = len(list(Path(str(root_path) + '/raw/Offshore/').rglob('*.jpg')))  # Get total number of image pairs

# Preprocess raw images (greyscale, resize, rename)
src_paths = {'x_path': str(root_path) + '/raw/Offshore/',
             'y_path': str(root_path) + '/raw/NearShore/'}
temp_paths = {'x_path': str(root_path) + '/interim/Offshore/',
              'y_path': str(root_path) + '/interim/NearShore/'}

os.makedirs(str(root_path) + '/interim')
os.makedirs(str(root_path) + '/interim/Offshore')
os.makedirs(str(root_path) + '/interim/NearShore')

for path in src_paths:
    for i in range(0, numSamples):
        img = cv.imread(src_paths[path] + str(i) + ".jpg", 0)
        img_scaled = cv.resize(img, (64, 64), cv.INTER_AREA)
        cv.imwrite(temp_paths[path] + str(i).zfill(5) + ".jpg", img_scaled)

# ----------------------------------------------------------------
# CLUSTERING
img2vec = Img2Vec(cuda=True if torch.cuda.is_available() else False)

DIR_PATH = str(root_path) + "/interim/Offshore"

project_name = str(args['name'])
embedding_path = f"data/clustered/embeddings/{project_name}.pt"
clusters_directory = f"data/clustered/clusters/{project_name}"

# Create required directories
required_dirs = ["data/clustered/embeddings", "data/clustered/clusters"]
for dir in required_dirs:
    utils.create_dir(dir)
utils.create_dir(clusters_directory)

# Get image data paths and read images
images = utils.read_images_from_directory(DIR_PATH)
images = images[0:int(int(args['s']) / 0.8)]
pil_images = utils.read_with_pil(images)

# Get embeddings
vec = img2vec.get_vec(pil_images, tensor=True)
utils.save_embeddings(vec, embedding_path)

# Embeddings -> PCA
pca_embeddings = clustering.calculate_pca(embeddings=vec, dim=2)

# PCA -> KMeans
clf = KMeansConstrained(
    n_clusters=args['c'],
    size_min=args['s'] // 0.8 // 40,        # e.g., for -s = 2400, total images = 3000, min. size = 75
)
clf.fit_predict(pca_embeddings)
centroid = clf.cluster_centers_
labels = clf.labels_

# Save random sample clusters
for label_number in tqdm(range(args['c'])):
    label_mask = labels == label_number
    label_images = list(compress(pil_images, label_mask))
    utils.create_image_grid(label_images, project_name, label_number)

    path_images = list(compress(images, label_mask))
    target_directory = f"data/clustered/clusters/{project_name}/cluster_{label_number}"
    utils.create_dir(target_directory)

    # Copy images into separate directories
    for img_path in path_images:
        shutil.copy2(
            img_path,
            target_directory,
        )

# Initialize the class object
kmeans = KMeans(n_clusters=args['c'])

# Predict cluster labels
clf = KMeansConstrained(
    n_clusters=args['c'],
    size_min=args['s'] // 0.8 // 40,        # e.g., for -s = 2400, total images = 3000, min. size = 75
)

# Get unique labels
label = clf.fit_predict(pca_embeddings)
centroids = clf.cluster_centers_
u_labels = np.unique(label)

# Plot results:
plt.close()
for i in u_labels:
    plt.scatter(pca_embeddings[label == i, 0], pca_embeddings[label == i, 1], label=i)
plt.scatter(centroids[:, 0], centroids[:, 1], s=80, color='k')
# plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.tight_layout()
plt.savefig("data/clustered/KMeans.jpg")
plt.show()
plt.close()


# ----------------------------------------------------------------
# Split dataset into training/test sets
if not os.path.exists(str(root_path) + '/processed'):
    os.makedirs(str(root_path) + '/processed/x_train')
    os.makedirs(str(root_path) + '/processed/x_val')
    os.makedirs(str(root_path) + '/processed/x_test')
    os.makedirs(str(root_path) + '/processed/y_train')
    os.makedirs(str(root_path) + '/processed/y_val')
    os.makedirs(str(root_path) + '/processed/y_test')

x_dst_paths = {'x_train': str(root_path) + '/processed/x_train/',
               'x_val': str(root_path) + '/processed/x_val/',
               'x_test': str(root_path) + '/processed/x_test/'}

y_dst_paths = {'y_train': str(root_path) + '/processed/y_train/',
               'y_val': str(root_path) + '/processed/y_val/',
               'y_test': str(root_path) + '/processed/y_test/'}

x_dst_keys = list(x_dst_paths)
y_dst_keys = list(y_dst_paths)

lengths = {'train': args['s'],
           'val': args['s'] // 8,
           'test': args['s'] // 8}

iterLengths = {"train": lengths["train"],
               "val": lengths["train"] + lengths["val"],
               "test": lengths["train"] + lengths["val"] + lengths["test"]}

numSamples = lengths["train"] + lengths["val"] + lengths["test"]    # total dataset size calculated from training set

print("# of pairs:\t", numSamples, "\nDataset split:\t", lengths)

# Can change these to hardcoded values
train_percent = args['s'] // 0.8 // 60      # e.g., // 60 = 50 images
val_percent = args['s'] // 0.8 // 200       # e.g., // 200 = 15 images
test_percent = args['s'] // 0.8 // 300      # e.g., // 300 = 10 images

print("Minimum cluster size:\t", train_percent + val_percent + test_percent)
print("# of samples per cluster (train):\t", train_percent)
print("# of samples per cluster (val):\t\t", val_percent)
print("# of samples per cluster (test):\t", test_percent)

d = f'data/clustered/clusters/{project_name}'
subdirs = [os.path.join(d, o) for o in os.listdir(d) if os.path.isdir(os.path.join(d, o))]

for dir in subdirs:
    files = [f for f in os.listdir(dir) if os.path.isfile(dir + '/' + f)]
    to_copy = random.sample(files, int(train_percent + val_percent + test_percent))
    for i in range(int(train_percent + val_percent + test_percent)):
        if i < int(train_percent):
            shutil.copy(os.path.join(dir, to_copy[i]), x_dst_paths['x_train'])
        elif int(train_percent) <= i < int(train_percent + val_percent):
            shutil.copy(os.path.join(dir, to_copy[i]), x_dst_paths['x_val'])
        elif int(train_percent + val_percent) <= i < int(train_percent + val_percent + test_percent):
            shutil.copy(os.path.join(dir, to_copy[i]), x_dst_paths['x_test'])

# copy corresponding near shore data into y_directories
d = f'data/processed'
subdirs = [os.path.join(d, o) for o in os.listdir(d) if os.path.isdir(os.path.join(d, o))]
x_train_files = [f for f in os.listdir("data/processed/x_train") if os.path.isfile("data/processed/x_train" + '/' + f)]
x_val_files = [f for f in os.listdir("data/processed/x_val") if os.path.isfile("data/processed/x_val" + '/' + f)]
x_test_files = [f for f in os.listdir("data/processed/x_test") if os.path.isfile("data/processed/x_test" + '/' + f)]

for f in x_train_files:
    shutil.copy(os.path.join("data/interim/NearShore", f), y_dst_paths['y_train'])
for f in x_val_files:
    shutil.copy(os.path.join("data/interim/NearShore", f), y_dst_paths['y_val'])
for f in x_test_files:
    shutil.copy(os.path.join("data/interim/NearShore", f), y_dst_paths['y_test'])

# ----------------------------------------------------------------
# Calculate the mean and standard deviation of training samples
files = list(Path(str(root_path) + '/processed/x_train/').rglob('*.jpg'))  # Get number of image pairs in x_train
numTrainSamples = len(files)

mean = np.array([0.])
stdTemp = np.array([0.])
std = np.array([0.])

for i in tqdm(range(numTrainSamples)):
    im = cv.imread(str(files[i]), 0)
    im = im.astype(float) / 255.

    for j in range(1):
        mean[j] += np.mean(im[:, j])

mean = (mean / numTrainSamples)

for i in tqdm(range(numTrainSamples)):
    im = cv.imread(str(files[i]), 0)
    im = im.astype(float) / 255.
    for j in range(1):
        stdTemp[j] += ((im[:, j] - mean[j]) ** 2).sum() / (im.shape[0] * im.shape[1])

std = np.sqrt(stdTemp / numTrainSamples)

print("Mean:\t", mean)
print("Std:\t", std)

# Calculated values:
# Mean:     [0.1176469]
# Std:      [0.00040002]
