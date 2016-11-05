#!/usr/bin/env python

from keras.models import load_model
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys
import os

files_dir = os.path.dirname(__file__)
dae_path = os.path.join(files_dir, "./dae_model.h5")

def saltpepper(images, salt=0.1, pepper=0.1):
    ret = np.copy(images)
    imagesize = images.shape[1]*images.shape[2]
    r = np.random.random(ret.shape)
    ret[r < salt] = 1
    ret[(salt < r) & (r < pepper+salt)] = 0
    return ret

# load images
if len(sys.argv) != 2:
    print("./predict_dae.py [png file path]")
    exit(-1)

file_path = sys.argv[1]

if not os.path.exists(file_path):
    print("Not found:", file_path)
    exit(-1)

image = Image.open(file_path)
image = np.array(image, dtype=np.float32).reshape([1, 32, 32, 1]) / 255
images = np.concatenate([
    saltpepper(image, 0, 0),
    saltpepper(image, 0.05, 0.05),
    saltpepper(image, 0.1, 0.1),
    saltpepper(image,0.2, 0.2),
    saltpepper(image, 0.3, 0.3)
]).reshape(-1,32,32,1)

# load model
if not os.path.exists(dae_path):
    print("Model file not found:", dae_path)
    exit(-1)

# predict
dae = load_model(dae_path)
p = dae.predict(images)

plt.figure(figsize=(16, 4))
plt.gray()
n = 5
for i in range(n):
    ax = plt.subplot(2, n, i+1)
    ax.set_title("Noise:{0}%".format([0, 10, 20, 40, 60][i]))
    plt.imshow(images[i].reshape(32, 32))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i + n + 1)
    plt.imshow(p[i].reshape(32, 32))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()