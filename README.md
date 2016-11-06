[http://qiita.com/t-ae/items/8424c7b788ee8c5c2109](http://qiita.com/t-ae/items/8424c7b788ee8c5c2109)

# Setup
1. Put ETL6 and ETL7 directories on root directory ([ETL Character Database](http://etlcdb.db.aist.go.jp/)).
1. Run `./extract.py`. npy files will be created under "data" directory.
1. Run `./create_train_data.npy`. dataset will be created under "train" directory.

# Train & Predict

## Simple CNN Model
1. Run `./simple_cnn/train.py`.
1. Run `./simple_cnn/predict.py data/test_XX.npy` to generate abyss letter for XX.

- `tensorboard --logdir=/tmp/abyss_logs/simple_cnn`

## Autoencoder+CNN Model
1. Run `./autoencoder/train_autoencoder.py`.
1. Run `./autoencoder/train_generator.py`.
1. Run `./autoencoder/predict_generator.py data/test_XX.npy` to generate abyss letter for XX.

- Run `./autoencoder/predict_autoencoder.py data/test_XX.npy` to show autoencoder result for XX.
- `tensorboard --logdir=/tmp/abyss_logs/autoencoder`
- `tensorboard --logdir=/tmp/abyss_logs/generator`

# Directories

- **data**: extracted npy files
- **abyss_letters**: png images of abyss letters
- **train_data**: compounded train datas

# Characters
## Hiragana training chars
I, TO, HE, MO, YU, RI, N, - 
## Katakana training chars
 A,  U,  O,  
KA, KI, KU, KE, KO,  
SI, SU, SO,  
TA, TI, TU, TE,  
NA, NI, NE, NO,  
HA, HE,  
MA, MI, ME, MO,  
RA, RI, RU, RE, RO,  
WA, N, -  
## Target chars to estimate
SE, NU, HO