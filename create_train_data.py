#!/usr/bin/env python

import os
import glob
import numpy as np
from PIL import Image

files_dir = os.path.dirname(__file__)

size = 32

# create train data
abyss_letter_dir = os.path.join(files_dir, "./abyss_letters/*.png")
data_dir = os.path.join(files_dir, "./data")

X_train = np.array([])
y_train = np.array([])
X_val = np.array([])
y_val = np.array([])

for f in glob.glob(abyss_letter_dir):
    char = os.path.splitext(os.path.basename(f))[0]
    print(char)
    image = Image.open(f)
    abyss_image = np.array(image, dtype=np.float32).ravel() / 255.0
    
    data_path = os.path.join(data_dir, "./train_{0}.npy".format(char))
    data = np.load(data_path)

    vals = data.shape[0] // 10
    trains = data.shape[0] - vals

    X_train = np.concatenate([X_train, data[vals:].ravel()])
    y_train = np.concatenate([y_train] + [abyss_image.ravel()]*trains)
    X_val = np.concatenate([X_val, data[:vals].ravel()])
    y_val = np.concatenate([y_val] + [abyss_image.ravel()]*vals)

X_train = X_train.reshape([-1, size, size, 1])
y_train = y_train.reshape([-1, size, size, 1])
X_val= X_val.reshape([-1, size, size, 1])
y_val = y_val.reshape([-1, size, size, 1])

print("train X: {0}, y: {1}".format(X_train.shape, y_train.shape))
print("val   X: {0}, y: {1}".format(X_val.shape, y_val.shape))

np.save(os.path.join(files_dir, "./train/X_train.npy"), X_train)
np.save(os.path.join(files_dir, "./train/y_train.npy"), y_train)
np.save(os.path.join(files_dir, "./train/X_val.npy"), X_val)
np.save(os.path.join(files_dir, "./train/y_val.npy"), y_val)