#!/usr/bin/env python

from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

files_dir = os.path.dirname(__file__)
autoencoder_path = os.path.join(files_dir, "./autoencoder_model.h5")

# load images
if len(sys.argv) != 2:
    print("./predict_autoencoder.py [file path]")
    exit(-1)

file_path = sys.argv[1]

if not os.path.exists(file_path):
    print("Not found:", file_path)
    exit(-1)

images = np.load(file_path)

# load model
if not os.path.exists(autoencoder_path):
    print("Model file not found:", autoencoder_path)
    exit(-1)

# predict
autoencoder = load_model(autoencoder_path)
p = autoencoder.predict(images)

plt.figure(figsize=(16, 4))
n = 5
for i in range(n):
    ax = plt.subplot(2, n, i+1)
    plt.imshow(images[i].reshape(32, 32))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i + n + 1)
    plt.imshow(p[i].reshape(32, 32))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()