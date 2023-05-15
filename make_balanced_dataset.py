from utils import clustering, clustering_utils as utils
from k_means_constrained import KMeansConstrained
import matplotlib.pyplot as plt
from itertools import compress
from pathlib import Path
from tqdm import tqdm
import numpy as np
import argparse
import random
import shutil
import torch
import os


parser = argparse.ArgumentParser(description="Preprocessing options")
parser.add_argument("-d", "--data", action="store", type=str, default='data', required=True,
                    help="path to parent directory of raw data folder")
parser.add_argument("-n", "--name", action="store", type=str, required=True,
                    help="dataset name")
parser.add_argument('--split', nargs='+', type=int, required=True,
                    help="number of train/val/test samples per cluster. e.g. --split 50 20 10")
parser.add_argument("-c", type=int, default=None, required=True,
                    help="number of clusters (classes)")

args = parser.parse_args()

project_name = args.name
clusters_directory = f"data/clustered/clusters/{project_name}"
utils.create_dir(clusters_directory)

root_path = Path(args.data)

x_data_path = "data/offshore_5yr.npz"
y_data_path = "data/nearshore_5yr.npz"

x_data = np.load(x_data_path)
y_data = np.load(y_data_path)

numSamples = (len(x_data.files))
print(f"# of x/y pairs:\t{numSamples}")
cluster_min = len(x_data.files) // args.c

temp_paths = {'x_path': str(root_path) + '/interim/Offshore/',
              'y_path': str(root_path) + '/interim/NearShore/'}

image_paths = list(Path(temp_paths['x_path']).rglob('*.npy'))

# os.makedirs(str(root_path) + '/interim')
# os.makedirs(temp_paths['x_path'])
# os.makedirs(temp_paths['y_path'])
#
# for i in range(0, numSamples):
#     x_img = x_data[str(i).zfill(5)].astype(np.float32)
#     np.save("data/interim/Offshore/"+str(i).zfill(5)+".npy", x_img)
#     y_img = y_data[str(i).zfill(5)].astype(np.float32)
#     np.save("data/interim/NearShore/" + str(i).zfill(5) + ".npy", y_img)

print("Created interim folders (.npy) files")
print("Converting images -> vectors")

x_vec = [None] * numSamples
y_vec = [None] * numSamples

for i in range(0, numSamples):
    x_vec[i] = torch.from_numpy(x_data[str(i).zfill(5)].astype(np.float32))

x_data.close()
y_data.close()

x_stack = torch.stack(x_vec)
x_flat = x_stack.flatten(start_dim=1)

# Embeddings -> PCA
print("Doing PCA...")
pca_embeddings = clustering.calculate_pca(embeddings=x_flat, dim=2)
print("PCA done")

# PCA -> KMeans
print(f"Doing Constrained KMeans: Minimum cluster size = {cluster_min}")
clf = KMeansConstrained(n_clusters=args.c, size_min=cluster_min)
label = clf.fit_predict(pca_embeddings)
centroid = clf.cluster_centers_
labels = clf.labels_
print("Clustering done")

print("Saving .npy files to relevant clusters")
for label_number in tqdm(range(args.c)):
    label_mask = labels == label_number
    label_images = list(compress(x_stack, label_mask))
    utils.create_image_grid(label_images, project_name, label_number)

    path_images = list(compress(image_paths, label_mask))
    target_directory = f"data/clustered/clusters/{project_name}/cluster_{label_number}"
    utils.create_dir(target_directory)

    # Copy images into separate directories
    for img_path in path_images:
        shutil.copy2(
            img_path,
            target_directory,
        )

print("Saved")
# Get unique labels
label = clf.fit_predict(pca_embeddings)
centroids = clf.cluster_centers_
u_labels = np.unique(label)

plt.close()
print("Plotting results...")
for i in u_labels:
    plt.scatter(pca_embeddings[label == i, 0], pca_embeddings[label == i, 1], label=i, s=20)
plt.scatter(centroids[:, 0], centroids[:, 1], s=80, color='k')
plt.tight_layout()
plt.savefig("data/clustered/KMeans.png")
plt.close()


print("Splitting data into train/validation/test sets...")

lengths = {'train': args.split[0] * args.c,
           'val': args.split[1] * args.c,
           'test': args.split[2] * args.c}

iterLengths = {"train": lengths["train"],
               "val": lengths["train"] + lengths["val"],
               "test": lengths["train"] + lengths["val"] + lengths["test"]}

numSamples = lengths["train"] + lengths["val"] + lengths["test"]

print("# of pairs:\t", numSamples, "\nDataset split:\t", lengths)

x_train_data = []
x_train_ids = []
x_validation_data = []
x_validation_ids = []
x_test_data = []
x_test_ids = []

d = f'data/clustered/clusters/{project_name}'
sub_dirs = [os.path.join(d, o) for o in os.listdir(d) if os.path.isdir(os.path.join(d, o))]

for sub_dir in sub_dirs:
    files = [f for f in os.listdir(sub_dir) if os.path.isfile(sub_dir + '/' + f)]
    to_copy = random.sample(files, args.split[0] + args.split[1] + args.split[2])
    for i in range(numSamples):
        if i < int(args.split[0]):
            x_train_data.append(np.load(sub_dir + "/" + to_copy[i]).astype(np.float32))
            x_train_ids.append(to_copy[i].split('.')[0])
        elif int(args.split[0]) <= i < int(args.split[0] + args.split[1]):
            x_validation_data.append(np.load(sub_dir + "/" + to_copy[i]).astype(np.float32))
            x_validation_ids.append(to_copy[i].split('.')[0])
        elif int(args.split[0] + args.split[1]) <= i < int(args.split[0] + args.split[1] + args.split[2]):
            x_test_data.append(np.load(sub_dir + "/" + to_copy[i]).astype(np.float32))
            x_test_ids.append(to_copy[i].split('.')[0])

x_train_dict = dict(zip(x_train_ids, x_train_data))
np.savez_compressed('data/x_train', **x_train_dict)

x_validation_dict = dict(zip(x_validation_ids, x_validation_data))
np.savez_compressed('data/x_val', **x_validation_dict)

x_test_dict = dict(zip(x_test_ids, x_test_data))
np.savez_compressed('data/x_test', **x_test_dict)

print("Offshore data split successfully")

y_train_data = []
y_train_ids = []
y_validation_data = []
y_validation_ids = []
y_test_data = []
y_test_ids = []

with np.load("data/nearshore_5yr.npz") as y_data:
    for i in enumerate(list(x_train_ids)):
        y_train_ids.append(i[-1])
        y_train_data.append(y_data[i[-1].zfill(5)])
    for i in enumerate(list(x_validation_ids)):
        y_validation_ids.append(i[-1])
        y_validation_data.append(y_data[i[-1].zfill(5)])
    for i in enumerate(list(x_test_ids)):
        y_test_ids.append(i[-1])
        y_test_data.append(y_data[i[-1].zfill(5)])

y_train_dict = dict(zip(y_train_ids, y_train_data))
np.savez_compressed('data/y_train', **y_train_dict)

y_validation_dict = dict(zip(y_validation_ids, y_validation_data))
np.savez_compressed('data/y_val', **y_validation_dict)

y_test_dict = dict(zip(y_test_ids, y_test_data))
np.savez_compressed('data/y_test', **y_test_dict)

print("Corresponding near shore data split successfully")
print("Done")
