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
decoder_path = os.path.join(files_dir, "./decoder_model.h5")
dae_path = os.path.join(files_dir, "./dae_model.h5")
tensorboard_logdir = "/tmp/abyss_logs/dae"

y_train = np.load(os.path.join(files_dir, "../train_data/y_train.npy"))
y_val = np.load(os.path.join(files_dir, "../train_data/y_val.npy"))

def saltpepper(images, salt=0.1, pepper=0.1):
    ret = np.copy(images)
    imagesize = images.shape[1]*images.shape[2]
    r = np.random.random(ret.shape)
    ret[r < salt] = 1
    ret[(salt < r) & (r < pepper+salt)] = 0
    return ret

y_train_noise = saltpepper(y_train)
y_val_noise = saltpepper(y_val)


if os.path.exists(dae_path):
    print("load models & fine tuning")
    dae = load_model(dae_path)
    decoder = dae.layers[1]

    sgd = SGD(lr=1e-4, momentum=0.9)
    dae.compile(optimizer=sgd, loss='binary_crossentropy')

    dae.fit(y_train_noise, y_train, 
                    nb_epoch=100,
                    batch_size=128,
                    validation_data=(y_val_noise, y_val),
                    callbacks=[
                        EarlyStopping(patience=1)
                    ])
    
else:

    encoder = Sequential([
        Convolution2D(16, 5, 5, border_mode='same', activation='relu', input_shape=[32, 32, 1]),
        MaxPooling2D(border_mode='same'), #16x16
        Convolution2D(32, 5, 5, border_mode='same', activation='relu'),
        MaxPooling2D(border_mode='same'), #8x8
        Convolution2D(8, 3, 3, border_mode='same', activation='relu'),
        MaxPooling2D(border_mode='same') #4x4
    ])

    decoder = Sequential([
        Convolution2D(8, 3, 3, border_mode='same', activation='relu', input_shape=[4, 4, 8]),
        UpSampling2D(), # 8x8
        Convolution2D(16, 3, 3, border_mode='same', activation='relu'),
        UpSampling2D(), # 16x16
        Convolution2D(32, 5, 5, border_mode='same', activation='relu'),
        UpSampling2D(), # 32x32
        Convolution2D(1, 5, 5, border_mode='same', activation='sigmoid')
    ])

    dae = Sequential()
    dae.add(encoder)
    dae.add(decoder)

    dae.compile(optimizer='adam', loss='binary_crossentropy')

    dae.fit(y_train_noise, y_train, 
                    nb_epoch=100,
                    batch_size=128,
                    validation_data=(y_val_noise, y_val),
                    callbacks=[
                        EarlyStopping(patience=1),
                        TensorBoard(log_dir=tensorboard_logdir)
                    ])

dae.save(dae_path)
decoder.save(decoder_path)