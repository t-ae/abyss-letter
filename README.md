# Setup
1. Put ETL6 and ETL7 directories in this directory.
1. Run `./extract.py`. npy files will be created under "data" directory.
1. Run `./create_train_data.npy`. dataset will be created under "train" directory.

# Simple CNN Model
1. Run `./simple_cnn/train.py`.
1. Run `./simple_cnn/predict.py data/test_XX.npy` to generate abyss letter for XX.

- `tensorboard --logdir=/tmp/abyss_logs/simple_cnn`

# Autoencoder+CNN Model
1. Run `./autoencoder/train_autoencoder.py`.
1. Run `./autoencoder/train_generator.py`.
1. Run `./autoencoder/predict_generator.py data/test_XX.npy` to generate abyss letter for XX.

- Run `./autoencoder/predict_autoencoder.py data/test_XX.npy` to show autoencoder result for XX.
- `tensorboard --logdir=/tmp/abyss_logs/autoencoder`
- `tensorboard --logdir=/tmp/abyss_logs/generator`