#!/usr/bin/env python

from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# load images
if len(sys.argv) != 2:
    print("./predict.py [file path]")
    exit(-1)

file_path = sys.argv[1]

if not os.path.exists(file_path):
    print("Not found:", file_path)
    exit(-1)

images = np.load(file_path)
images = images[:, ::2, ::2].reshape([-1, 32, 32, 1])

# load model
model_path = os.path.join(os.path.dirname(__file__), "./abyss_model.h5")

if not os.path.exists(model_path):
    print("Model file not found:", model_path)
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