#!/usr/bin/env python

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) != 2:
    print("./etl_viewer.py [npy file path]")
    exit(-1)

file_path = sys.argv[1]

if not os.path.exists(file_path):
    print("Not found:", file_path)
    exit(-1)

images = np.load(file_path)

num = 5

plt.gray()
for i in range(0, len(images), num):
    fig, axes = plt.subplots(ncols=num)
    for j in range(num):
        if num*i+j >= len(images):
            axes[j].axis('off')
            continue
        axes[j].imshow(images[num*i+j].reshape([32, 32]))
        axes[j].get_xaxis().set_visible(False)
        axes[j].get_yaxis().set_visible(False)
    plt.pause(1)