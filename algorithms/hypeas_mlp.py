# -*- coding: utf-8 -*-
import keras_tuner as kt
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import os, sys
import pandas as pd
import numpy as np

# Argument validation
if len(sys.argv) < 2:
    print("Usage: python hyperas_mlp.py <filename>")
    sys.exit(1)

filename = str(sys.argv[1])

if not os.path.exists(filename):
    print(f"Error: file '{filename}' not found.")
    sys.exit(1)

# ── Data preparation ──────────────────────────────────────────────────────────
# good practice to set the seed state, starting point
np.random.seed(7)

dataset = pd.read_csv(filename, usecols=[1], header=None)
dataset = dataset.values #convert to the array
dataset = dataset.astype('float32') # convert to float

# lenth of our data set
training_size = int(len(dataset)*0.67)
#testing_size = len(dataset)-training_size

# split the data set
train, test = dataset[0:training_size,:], dataset[training_size:len(dataset),:]

# one time step to the future
lookback = 1

#dataX, dataY = [], [] # create 2 empty list
dataX_train, dataY_train = [], [] # create 2 empty list
dataX_test, dataY_test = [], [] # create 2 empty list

# Fix: iterate over train/test splits separately, and move np.array() outside loop
for i in range(len(train)-lookback-1):
    a = train[i:(i+lookback),0]
    dataX_train.append(a)
    dataY_train.append(train[i+lookback,0]) # get the next value

X_train, Y_train = np.array(dataX_train), np.array(dataY_train)

for i in range(len(test)-lookback-1):
    a = test[i:(i+lookback),0]
    dataX_test.append(a)
    dataY_test.append(test[i+lookback,0]) # get the next value

X_test, Y_test = np.array(dataX_test), np.array(dataY_test)

# ── Model builder for Keras Tuner ─────────────────────────────────────────────
def build_model(hp):
    model = Sequential()
    model.add(Dense(
        hp.Choice('dense_l1', [30, 60, 80]),
        input_dim=lookback,
        activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(
        hp.Choice('dense_l2', [10, 30, 60]),
        activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))  # output layer, no activation for regression

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    return model

# ── Hyperparameter search ─────────────────────────────────────────────────────
tuner = kt.BayesianOptimization(    # BayesianOptimization aprende com tentativas anteriores
    build_model,
    objective=kt.Objective('val_mae', direction='min'),
    max_trials=5,
    directory='tuner_results',
    project_name='mlp_search',
    overwrite=True
)

tuner.search(
    X_train, Y_train,
    epochs=200,          # usa o maior valor como limite; EarlyStopping para antes se convergir
    batch_size=32,       # fixo aqui; otimize dentro do build_model se quiser variar
    validation_data=(X_test, Y_test),
    verbose=2,
    callbacks=[kt.callbacks.EarlyStopping(monitor='val_mae', patience=15)]
)

# ── Results ───────────────────────────────────────────────────────────────────
best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
best_model = tuner.get_best_models(num_models=1)[0]

print("Evalutation of best performing model:")
print(best_model.evaluate(X_test, Y_test))
print('')
print("Best performing model chosen hyper-parameters:")
print(best_hp.values)
print('')
print(filename)

LAYER1     = best_hp.get('dense_l1')
LAYER2     = best_hp.get('dense_l2')
BATCH_SIZE = best_hp.get('batch_size')
NB_EPOCHS  = best_hp.get('nb_epoch')

print('BATCH_SIZE =', BATCH_SIZE)
print('NB_EPOCHS =',  NB_EPOCHS)
print('LAYER1 =',     LAYER1)
print('LAYER2 =',     LAYER2)
print('')
print("python mlp.py %s %d %d %d %d" % (filename, BATCH_SIZE, NB_EPOCHS, LAYER1, LAYER2))

# Save into a file the command of executions and its scores
with open("MLP_hyperparameters.txt", "a", encoding="utf-8") as f:   # Fix: use with block
    f.write("\npython mlp.py %s %d %d %d %d" % (filename, BATCH_SIZE, NB_EPOCHS, LAYER1, LAYER2))