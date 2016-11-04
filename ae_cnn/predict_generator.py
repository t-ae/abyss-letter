#!/usr/bin/env python

from keras.models import Sequential, load_model
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

files_dir = os.path.dirname(__file__)
generator_path = os.path.join(files_dir, "./generator_model.h5")

# load images
if len(sys.argv) != 2:
    print("./predict_generator.py [file path]")
    exit(-1)

file_path = sys.argv[1]

if not os.path.exists(file_path):
    print("Not found:", file_path)
    exit(-1)

images = np.load(file_path)

# load model
if not os.path.exists(generator_path):
    print("Model file not found:", generator_path)
    exit(-1)

# predict
generator = load_model(generator_path)
p = generator.predict(images)

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