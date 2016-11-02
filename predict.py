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

# load model
model_path = os.path.join(os.path.dirname(__file__), "./abyss_model.h5")

if not os.path.exists(model_path):
    print("Model file not found:", model_path)
    exit(-1)

# predict
model = load_model(model_path)
p = model.predict(images)

