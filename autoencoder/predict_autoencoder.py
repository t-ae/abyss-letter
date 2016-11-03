#!/usr/bin/env python

from keras.models import Model
from keras.layers import Input
from keras.layers.convolutional import Convolution2D, UpSampling2D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import EarlyStopping, TensorBoard
import numpy as np
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
if not os.path.exists(model_path):
    print("Model file not found:", autoencoder_path)
    exit(-1)

# predict
model = load_model(model_path)
p = model.predict(images)

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