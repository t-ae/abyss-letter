#!/usr/bin/env python

from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Reshape
from keras.layers.advanced_activations import ELU
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, TensorBoard
import numpy as np
import os

files_dir = os.path.dirname(__file__)
encoder_path = os.path.join(files_dir, "../ae_cnn/encoder_model.h5")
decoder_path = os.path.join(files_dir, "./decoder_model.h5")
dae_generator_path = os.path.join(files_dir, "./dae_generator_model.h5")
tensorboard_logdir = "/tmp/abyss_logs/dae_generator"


X_train = np.load(os.path.join(files_dir, "../train_data/X_train.npy"))
y_train = np.load(os.path.join(files_dir, "../train_data/y_train.npy"))
X_val = np.load(os.path.join(files_dir, "../train_data/X_val.npy"))
y_val = np.load(os.path.join(files_dir, "../train_data/y_val.npy"))



# create model
if os.path.exists(dae_generator_path):
    print("load model & fine tuning")
    model = load_model(dae_generator_path)

    sgd = SGD(lr=1e-4, momentum=0.9)
    model.compile(optimizer=sgd, loss='binary_crossentropy')

    model.fit(X_train, y_train,
        nb_epoch=100,
        batch_size=128,
        validation_data=(X_val, y_val),
        callbacks=[
            EarlyStopping(patience=1)
        ])
else:
    # load encoder
    if not os.path.exists(encoder_path):
        print("Encoder model file not found")
        exit(-1)
    encoder = load_model(encoder_path)
    encoder.trainable = False
    for layer in encoder.layers:
        layer.trainable = False

    # load decoder
    if not os.path.exists(decoder_path):
        print("Decoder model file not found")
        exit(-1)
    decoder = load_model(decoder_path)
    decoder.trainable = False
    for layer in decoder.layers:
        layer.trainable = False

    dae_generator = Sequential([
        Flatten(input_shape=[4, 4, 16]),
        Dense(512),
        ELU(),
        Dense(512),
        ELU(),
        Dense(4*4*8),
        ELU(),
        Reshape([4, 4, 8])
    ])

    model = Sequential([encoder, dae_generator, decoder])
    model.compile(optimizer='adam', loss='binary_crossentropy')

    model.fit(X_train, y_train,
        nb_epoch=100,
        batch_size=128,
        validation_data=(X_val, y_val),
        callbacks=[
            EarlyStopping(patience=1),
            TensorBoard(log_dir=tensorboard_logdir)
        ])

model.save(dae_generator_path)