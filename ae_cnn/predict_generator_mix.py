#!/usr/bin/env python

from keras.models import Sequential, load_model
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

files_dir = os.path.dirname(__file__)
generator_path = os.path.join(files_dir, "./generator_model.h5")

# load images
if len(sys.argv) != 3:
    print("./predict_generator_mix.py [file path 1] [file path 2]")
    exit(-1)

file_path1 = sys.argv[1]
file_path2 = sys.argv[2]
if not os.path.exists(file_path1):
    print("Not found:", file_path1)
    exit(-1)
if not os.path.exists(file_path2):
    print("Not found:", file_path2)
    exit(-1)

images1 = np.load(file_path1)
images2 = np.load(file_path2)

# load model
if not os.path.exists(generator_path):
    print("Model file not found:", generator_path)
    exit(-1)

# predict
generator = load_model(generator_path)

encoder = generator.layers[0]
generator = generator.layers[1]

encoded1 = encoder.predict(images1)
encoded2 = encoder.predict(images2)

mixed1 = 0.3*encoded1+0.7*encoded2
mixed2 = 0.5*encoded1+0.5*encoded2
mixed3 = 0.7*encoded1+0.3*encoded2

p1 = generator.predict(mixed1)
p2 = generator.predict(mixed2)
p3 = generator.predict(mixed3)
plt.figure()
plt.gray()
n = 5
for i in range(n):
    ax = plt.subplot(5, n, i+1)
    plt.imshow(images1[i].reshape(32, 32))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(5, n, i + n + 1)
    plt.imshow(images2[i].reshape(32, 32))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(5, n, i + 2*n + 1)
    ax.set_title("3:7")
    plt.imshow(p1[i].reshape(32, 32))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(5, n, i + 3*n + 1)
    ax.set_title("5:5")
    plt.imshow(p2[i].reshape(32, 32))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(5, n, i + 4*n + 1)
    ax.set_title("7:3")
    plt.imshow(p3[i].reshape(32, 32))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.tight_layout()
plt.show()