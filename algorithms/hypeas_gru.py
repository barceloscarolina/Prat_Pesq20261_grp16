# -*- coding: utf-8 -*-
import keras_tuner as kt
from keras.layers import GRU, Dense, Dropout
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import os, sys
import pandas as pd
import numpy as np

# Argument validation
if len(sys.argv) < 2:
    print("Usage: python hyperas_gru.py <filename>")
    sys.exit(1)

filename = str(sys.argv[1])

# ── Data preparation ──────────────────────────────────────────────────────────
dataset = pd.read_csv(filename, usecols=[1], header=None)
dataset = dataset.values.astype('float32')

training_size = int(len(dataset) * 0.67)
train, test = dataset[0:training_size], dataset[training_size:]

lookback = 1

def create_windows(data, lookback):
    dataX, dataY = [], []
    for i in range(len(data)-lookback-1):
        dataX.append(data[i:(i+lookback), 0])
        dataY.append(data[i+lookback, 0])   # get the next value
    return np.array(dataX), np.array(dataY)

X_train, Y_train = create_windows(train, lookback)
X_test,  Y_test  = create_windows(test,  lookback)

# scaling values for model
scaleX = MinMaxScaler()
scaleY = MinMaxScaler()

X_train = scaleX.fit_transform(X_train).reshape((-1,1,1))
Y_train = scaleY.fit_transform(Y_train.reshape(-1,1))
X_test  = scaleX.transform(X_test).reshape((-1,1,1))    # transform only, not fit_transform
Y_test  = scaleY.transform(Y_test.reshape(-1,1))

# ── Model builder for Keras Tuner ─────────────────────────────────────────────
def build_model(hp):
    #Create Keras model with hyperparameter choices dropped-in as needed.
    model = Sequential()
    model.add(GRU(
        units=hp.Choice('units_l1', [32, 128, 256]),
        return_sequences=True,
        input_shape=(1, 1)))
    model.add(Dropout(0.2))
    model.add(GRU(units=hp.Choice('units_l2', [16, 32, 64])))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    return model

# ── Hyperparameter search ─────────────────────────────────────────────────────
tuner = kt.RandomSearch(
    build_model,
    objective=kt.Objective('val_mae', direction='min'),
    max_trials=5,
    directory='tuner_results',
    project_name='gru_search'
)

tuner.search(
    X_train, Y_train,
    epochs=50,
    batch_size=130,
    validation_data=(X_test, Y_test),
    verbose=2
)

# ── Results ───────────────────────────────────────────────────────────────────
best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
best_model = tuner.get_best_models(num_models=1)[0]

print("Evaluation of best performing model:")
print(best_model.evaluate(X_test, Y_test))
print('')
print("Best performing model chosen hyper-parameters:")
print(best_hp.values)
print('')
print(filename)

LAYER1     = best_hp.get('units_l1')
LAYER2     = best_hp.get('units_l2')

print('LAYER1 =', LAYER1)
print('LAYER2 =', LAYER2)
print('')
print("python gru.py %s 130 50 %d %d" % (filename, LAYER1, LAYER2))

# Save into a file the command of executions and its scores
with open("gru_hyperparameters.txt", "a", encoding="utf-8") as f:
    f.write("\npython gru.py %s 130 50 %d %d" % (filename, LAYER1, LAYER2))