import numpy as np
import os
import cv2 as cv
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random
import shutil
import argparse


parser = argparse.ArgumentParser(description="Preprocessing root path")

parser.add_argument("-d", "--data", action="store", type=str, required=True,
                    help="path to parent directory of raw data folder")

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
# Split dataset into training/test sets
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

lengths = {'train': int((0.8 * 0.8 * numSamples)),  # 80% of 80% of numSamples:     12800
           'val': int((0.2 * 0.8 * numSamples)),  # 20% of 80% of numSamples:       3200
           'test': int((0.2 * numSamples))}  # 20% of numSamples:                   4000

print("# of pairs:\t", numSamples, "\nDataset split:\t", lengths)

iterLengths = {"train": lengths["train"],                                    # 12800
               "val": lengths["train"] + lengths["val"],                     # 16000
               "test": lengths["train"] + lengths["val"] + lengths["test"]}  # 20000

random.seed(42)
randomList = random.sample(range(0, numSamples), numSamples)  # randomised list from 0 -> numSamples

indices = {"train_ind": randomList[0:iterLengths["train"]],                  # 0 to 12800 random indices
           "val_ind": randomList[iterLengths["train"]:iterLengths["val"]],   # 12800 to 16000 random indices
           "test_ind": randomList[iterLengths["val"]:iterLengths["test"]]}   # 16000 to 20000 random indices

for path in temp_paths:                 # iterate once each offshore/near shore
    for j in zip(lengths, iterLengths, indices, range(3)):
        if path.startswith("x"):
            for i in range(lengths[j[0]]):
                shutil.copyfile(temp_paths[path] + str(indices[j[2]][i]).zfill(5) + '.jpg',
                                x_dst_paths[x_dst_keys[j[3]]] + str(indices[j[2]][i]).zfill(5) + '.jpg')
        elif path.startswith("y"):
            for i in range(lengths[j[0]]):
                shutil.copyfile(temp_paths[path] + str(indices[j[2]][i]).zfill(5) + '.jpg',
                                y_dst_paths[y_dst_keys[j[3]]] + str(indices[j[2]][i]).zfill(5) + '.jpg')

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
