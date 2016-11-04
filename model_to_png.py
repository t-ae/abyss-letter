#!/usr/bin/env python

from keras.models import load_model
from keras.utils.visualize_util import plot
import numpy as np
import os
import sys

# load model
if len(sys.argv) != 2:
    print("./model_to_png.py [h5 path]")
    exit(-1)

file_path = sys.argv[1]

if not os.path.exists(file_path):
    print("Not found:", file_path)
    exit(-1)

model = load_model(file_path)

png_file_name = os.path.basename(file_path) + ".png"
plot(model, show_shapes=True, to_file=png_file_name)
print("Save:", png_file_name)