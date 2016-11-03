#!/usr/bin/env python

from keras.models import Sequential, load_model
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D, UpSampling2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.advanced_activations import ELU
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, TensorBoard
import numpy as np
import os

files_dir = os.path.dirname(__file__)
encoder_path = os.path.join(files_dir, "./encoder_model.h5")
generator_path = os.path.join(files_dir, "./generator_model.h5")
tensorboard_logdir = "/tmp/abyss_logs/generator"


X_train = np.load(os.path.join(files_dir, "../train_data/X_train.npy"))
y_train = np.load(os.path.join(files_dir, "../train_data/y_train.npy"))
X_val = np.load(os.path.join(files_dir, "../train_data/X_val.npy"))
y_val = np.load(os.path.join(files_dir, "../train_data/y_val.npy"))

# load encoder
if not os.path.exists(encoder_path):
    print("Encoder model file not found")
    exit(-1)
encoder = load_model(encoder_path)
encoder.trainable = False

# load generator
if os.path.exists(generator_path):
    print("load model & fine tuning")
    generator = load_model(generator_path)

    model = Sequential([encoder, generator])

    sgd = SGD(lr=1e-4, momentum=0.9)
    model.compile(optimizer=sgd, loss='binary_crossentropy')

    model.fit(X_train, y_train,
        nb_epoch=0,
        batch_size=128,
        validation_data=(X_val, y_val),
        callbacks=[
            EarlyStopping(patience=3),
            TensorBoard(log_dir=tensorboard_logdir)
        ])
else:
    generator = Sequential([
        Convolution2D(16, 3, 3, border_mode='same', input_shape=[4, 4, 16]),
        ELU(),
        UpSampling2D(), # 8x8
        Convolution2D(32, 3, 3, border_mode='same'),
        ELU(),
        UpSampling2D(), # 16x16
        Convolution2D(16, 5, 5, border_mode='same'),
        ELU(),
        UpSampling2D(), # 32x32
        Convolution2D(1, 5, 5, border_mode='same')
    ])

    model = Sequential([encoder, generator])
    model.compile(optimizer='adam', loss='binary_crossentropy')

    model.fit(X_train, y_train,
        nb_epoch=0,
        batch_size=128,
        validation_data=(X_val, y_val),
        callbacks=[
            EarlyStopping(patience=3),
            TensorBoard(log_dir=tensorboard_logdir)
        ])

generator.save(generator_path)