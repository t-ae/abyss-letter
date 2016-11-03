#!/usr/bin/env python

from keras.models import Model
from keras.layers import Input
from keras.layers.convolutional import Convolution2D, UpSampling2D
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import EarlyStopping, TensorBoard
import numpy as np
import os

files_dir = os.path.dirname(__file__)
encoder_path = os.path.join(files_dir, "./encoder_model.h5")
autoencoder_path = os.path.join(files_dir, "./autoencoder_model.h5")
tensorboard_logdir = "/tmp/abyss_logs/autoencoder"

X_train = np.load(os.path.join(files_dir, "../train/X_train.npy"))
X_val = np.load(os.path.join(files_dir, "../train/X_val.npy"))

input = Input(shape=[32, 32, 1])
# encode
x = Convolution2D(16, 5, 5, border_mode='same', activation='relu')(input)
x = MaxPooling2D(border_mode='same')(x) #16x16
x = Convolution2D(32, 5, 5, border_mode='same', activation='relu')(x)
x = MaxPooling2D(border_mode='same')(x) #8x8
x = Convolution2D(16, 3, 3, border_mode='same', activation='relu')(x)
encoded = MaxPooling2D(border_mode='same')(x) #4x4
# decode
x = Convolution2D(16, 3, 3, border_mode='same', activation='relu')(encoded)
x = UpSampling2D()(x) # 8x8
x = Convolution2D(16, 3, 3, border_mode='same', activation='relu')(x)
x = UpSampling2D()(x) # 16x16
x = Convolution2D(32, 5, 5, border_mode='same', activation='relu')(x)
x = UpSampling2D()(x) # 32x32
decoded = Convolution2D(1, 5, 5, border_mode='same', activation='sigmoid')(x)

autoencoder = Model(input, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.fit(X_train, X_train, 
                nb_epoch=100,
                validation_data=(X_val, X_val)
                callbacks=[
                    EarlyStopping(patience=3),
                    TensorBoard(log_dir=tensorboard_logdir)
                ])

autoencoder.save(autoencoder_path)


encoder = Model(input, encoder)
encoder.save(encoder_path)

