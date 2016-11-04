#!/usr/bin/env python

import os
import glob
import numpy as np
from PIL import Image


# rotation angle of abyss letters
rotate_max = 15


files_dir = os.path.dirname(__file__)
size = 32


def transform_images(image, num):
    if rotate_max == 0:
        return [image]*num
    else:
        rotate = np.random.uniform(-rotate_max, rotate_max, num)
        return [image.rotate(rotate[i]) for i in range(num)]
    

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
    y_train_images = transform_images(image, trains)
    y_train = np.concatenate([y_train] + [np.array(image, dtype=np.float32).ravel() / 255.0 for image in y_train_images])
    
    X_val = np.concatenate([X_val, data[:vals].ravel()])
    y_val_images = transform_images(image, vals)
    y_val = np.concatenate([y_val] + [np.array(image, dtype=np.float32).ravel() / 255.0 for image in y_val_images])


X_train = X_train.reshape([-1, size, size, 1])
y_train = y_train.reshape([-1, size, size, 1])
X_val= X_val.reshape([-1, size, size, 1])
y_val = y_val.reshape([-1, size, size, 1])

print("train X: {0}, y: {1}".format(X_train.shape, y_train.shape))
print("val   X: {0}, y: {1}".format(X_val.shape, y_val.shape))

np.save(os.path.join(files_dir, "./train_data/X_train.npy"), X_train)
np.save(os.path.join(files_dir, "./train_data/y_train.npy"), y_train)
np.save(os.path.join(files_dir, "./train_data/X_val.npy"), X_val)
np.save(os.path.join(files_dir, "./train_data/y_val.npy"), y_val)