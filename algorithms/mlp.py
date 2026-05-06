# -*- coding: utf-8 -*-
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import mean_squared_error, r2_score
import math
import os, sys

def create_dataset(dataset, lookback=1):
    dataX, dataY = [], [] # create 2 empty list

    # go through the lenght of dataset, subtract the lookback and 1. 
    #2 steps before the end of dataset, because we predict 1 step to the future
    for i in range(len(dataset)-lookback-1):
        a = dataset[i:(i+lookback),0]
        dataX.append(a)
        dataY.append(dataset[i+lookback,0]) # get the next value
    return np.array(dataX), np.array(dataY)

def save_model(model):
    filename = "data/" + "mlp" + ".h5"
    filename = "mlp" + ".h5"
    model.save(filename)

# receive the parameters by command line
# Argument validation
if len(sys.argv) < 6:
    print("Usage: python mlp.py <filename> <batch_size> <epochs> <layer1> <layer2>")
    print("Example: python mlp.py data.csv 32 150 64 32")
    sys.exit(1)

filename, BATCH_SIZE, NB_EPOCHS, LAYER1, LAYER2 = str(sys.argv[1]), int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]),int(sys.argv[5])

#perc = float(sys.argv[6])

if not os.path.exists(filename):
    print(f"Error: file '{filename}' not found.")
    sys.exit(1)

np.random.seed(7)
# Data preparation
dataset = pd.read_csv(filename, usecols=[1], header=None)
dataset = dataset.values #convert to the array
dataset = dataset.astype('float32') # convert to float

# length of our data set
training_size = int(len(dataset)*0.67)
testing_size = len(dataset)-training_size

# split the data set
train, test = dataset[0:training_size,:], dataset[training_size:len(dataset),:]
#print(len(train),len(test))

# one time step to the future
lookback = 1
trainX, trainY = create_dataset(train, lookback)
testX, testY = create_dataset(test, lookback)

#size = int(training_size * (perc /100))
#trainX = trainX[0:training_size - size:]
#strainY = trainY[0:training_size - size:]

# create the model
model_name = 'requests_MLP'
model=Sequential()
model.add(Dense(LAYER1, input_dim=lookback, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(LAYER2, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))

#Compilation and training
timer = datetime.datetime.now() 
start = datetime.datetime.now()
model.compile(loss='mean_squared_error', optimizer='adam') 

print("Compilation Time : ", datetime.datetime.now() - start)  # Fix: print() function
start = datetime.datetime.now() 

model.fit(trainX, trainY, epochs=NB_EPOCHS, batch_size=BATCH_SIZE, verbose=0) 
print("Training time : ", datetime.datetime.now() - start)     # Fix: print() function

#model.save("{}.h5".format(model_name))

# Making predictions
yhat = model.predict(trainX)
y_train = trainY

train_score = mean_squared_error(y_train, yhat)
print('Trainscore: %.2f MSE (%.2f RMSE)' % (train_score, math.sqrt(train_score)))
start = datetime.datetime.now() 
yhat = model.predict(testX)
y_test = testY
r2 = r2_score(y_test, yhat)

print("test time: ", datetime.datetime.now() - start)          # Fix: print() function
print("timer: ", datetime.datetime.now() - timer)              # Fix: print() function

# configuration executed
print('Filename:', filename)
print('BATCH_SIZE:', BATCH_SIZE)
print('NB_EPOCHS: ', NB_EPOCHS) 
print('LAYER1: ', LAYER1)
print('LAYER2: ', LAYER2) 

print('R2: ', r2)
test_score = mean_squared_error(y_test, yhat)
print('Testscore %.2f' % (math.sqrt(test_score)))
#print ('%.2f , %.2f ,' %(math.sqrt(score),perc)) + filename

# Save into a file the command of executions and its scores
with open("MLP_Execucoes.txt", "a", encoding="utf-8") as f:    # Fix: special char + with block
    f.write("\n python mlp.py %s %d %d %d %d - R2: %.2f, Trainscore: %.2f MSE (%.2f RMSE), Testscore: %.2f MSE (%.2f RMSE)" % (filename, BATCH_SIZE, NB_EPOCHS, LAYER1, LAYER2, r2, train_score, math.sqrt(train_score), test_score, math.sqrt(test_score))) 

#print 'actual,  predicted'
#for i in range(0, len(y_test)):
#  print('%.2f, %.2f' % (y_test[i],yhat[i+1]))

#plot executions
#plt.plot(yhat[-100:], label='Predicted')
#plt.plot(y_test[-100:], label='Current')
#plt.legend()
#plt.grid()

#plt.ylabel('Requests')
#plt.xlabel('Samples 15 min')
#plt.show()