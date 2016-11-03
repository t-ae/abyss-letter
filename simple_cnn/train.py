#!/usr/bin/env python

from keras.models import Sequential, load_model
from keras.layers.convolutional import Convolution2D, UpSampling2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.advanced_activations import ELU
from keras.callbacks import EarlyStopping, TensorBoard
import numpy as np
import os
import glob

files_dir = os.path.dirname(__file__)
model_path = os.path.join(files_dir, "./abyss_model.h5")
tensorboard_logdir = '/tmp/abyss_logs/simple_cnn'

X_train = np.load(os.path.join(files_dir, "../train_data/X_train.npy"))
y_train = np.load(os.path.join(files_dir, "../train_data/y_train.npy"))
X_val = np.load(os.path.join(files_dir, "../train_data/X_val.npy"))
y_val = np.load(os.path.join(files_dir, "../train_data/y_val.npy"))

# create model
if os.path.exists(model_path):
    print("load model")
    model = load_model(model_path)
else:
    model = Sequential([
        Convolution2D(64, 5, 5, border_mode='same', activation='relu', input_shape=[32, 32, 1]),
        MaxPooling2D(border_mode='same'), #16x16
        Convolution2D(128, 5, 5, border_mode='same', activation='relu'),
        MaxPooling2D(border_mode='same'), #8x8
        Convolution2D(32, 3, 3, border_mode='same', activation='relu'),
        MaxPooling2D(border_mode='same'), #4x4
        Convolution2D(64, 3, 3, border_mode='same'),
        ELU(),
        UpSampling2D(), # 8x8
        Convolution2D(128, 3, 3, border_mode='same'),
        ELU(),
        UpSampling2D(), # 16x16
        Convolution2D(64, 5, 5, border_mode='same'),
        ELU(),
        UpSampling2D(), # 32x32
        Convolution2D(1, 5, 5, border_mode='same', activation='sigmoid')
    ])

    model.compile(optimizer='adam', 
                loss='binary_crossentropy')

model.fit(X_train, y_train,
    nb_epoch=100,
    batch_size=128,
    validation_data=(X_val, y_val),
    callbacks=[
        EarlyStopping(patience=3),
        TensorBoard(log_dir=tensorboard_logdir)
    ])

model.save(model_path)