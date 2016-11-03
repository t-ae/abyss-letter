#!/usr/bin/env python

from keras.models import Sequential, load_model
from keras.layers import Input
from keras.layers.convolutional import Convolution2D, UpSampling2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, TensorBoard
import numpy as np
import os

files_dir = os.path.dirname(__file__)
encoder_path = os.path.join(files_dir, "./encoder_model.h5")
autoencoder_path = os.path.join(files_dir, "./autoencoder_model.h5")
tensorboard_logdir = "/tmp/abyss_logs/autoencoder"

X_train = np.load(os.path.join(files_dir, "../train_data/X_train.npy"))
X_val = np.load(os.path.join(files_dir, "../train_data/X_val.npy"))

if os.path.exists(autoencoder_path):
    print("load models & fine tuning")
    autoencoder = load_model(autoencoder_path)
    encoder = autoencoder.layers[0]

    sgd = SGD(lr=1e-4, momentum=0.9)
    autoencoder.compile(optimizer=sgd, loss='binary_crossentropy')

    autoencoder.fit(X_train, X_train, 
                    nb_epoch=100,
                    batch_size=128,
                    validation_data=(X_val, X_val),
                    callbacks=[
                        EarlyStopping(patience=3),
                        TensorBoard(log_dir=tensorboard_logdir)
                    ])
    
else:
    encoder = Sequential([
        Convolution2D(16, 5, 5, border_mode='same', activation='relu', input_shape=[32, 32, 1]),
        MaxPooling2D(border_mode='same'), #16x16
        Convolution2D(32, 5, 5, border_mode='same', activation='relu'),
        MaxPooling2D(border_mode='same'), #8x8
        Convolution2D(16, 3, 3, border_mode='same', activation='relu'),
        MaxPooling2D(border_mode='same') #4x4
    ])

    decoder = Sequential([
        Convolution2D(16, 3, 3, border_mode='same', activation='relu', input_shape=[4, 4, 16]),
        UpSampling2D(), # 8x8
        Convolution2D(16, 3, 3, border_mode='same', activation='relu'),
        UpSampling2D(), # 16x16
        Convolution2D(32, 5, 5, border_mode='same', activation='relu'),
        UpSampling2D(), # 32x32
        Convolution2D(1, 5, 5, border_mode='same', activation='sigmoid')
    ])

    autoencoder = Sequential()
    autoencoder.add(encoder)
    autoencoder.add(decoder)

    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    autoencoder.fit(X_train, X_train, 
                    nb_epoch=100,
                    batch_size=128,
                    validation_data=(X_val, X_val),
                    callbacks=[
                        EarlyStopping(patience=3),
                        TensorBoard(log_dir=tensorboard_logdir)
                    ])

autoencoder.save(autoencoder_path)

encoder.save(encoder_path)