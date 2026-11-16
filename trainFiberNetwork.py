# %%
import sys
from datetime import datetime
import tensorflow as tf
import numpy as np
import h5py
import keras
from sklearn.model_selection import train_test_split

from fiberNet import createArchitecturePBMLP, createArchitecturePBKAN, getGridCallback
from dataLoaders import loadData, saveDictionary, saveModelWeights

tf.keras.backend.clear_session()

fileName = sys.argv[1]
epochs = int( sys.argv[2] )
batchSize = int( sys.argv[3] )
validationSplit = float( sys.argv[4] )

timeStamp = datetime.now()
timeStamp = datetime.timestamp( timeStamp )
timeStamp = int( timeStamp )
timeStamp = str( timeStamp )

def scheduler(epoch, lr):
    if epoch < 5:
        return 0.001
    elif lr <= 0.000001:
        return 0.000001
    else:
        return lr * tf.math.exp(-0.1)

X, y = loadData( fileName )
print("y: ")
print( y.shape )

inputSize = X.shape[1]
outputSize = y.shape[1]

x_train, x_valid, y_train, y_valid = train_test_split( X, y, test_size = validationSplit, shuffle= True)

y_train = [ y_train[:, 0:2], y_train[:, 2:4], y_train[:,4] ]
y_valid = [ y_valid[:, 0:2], y_valid[:, 2:4], y_valid[:,4] ]

learningRateCallback = tf.keras.callbacks.LearningRateScheduler( scheduler )
stoppingCallback = tf.keras.callbacks.EarlyStopping( monitor = 'val_loss',
                                                     patience = 10, 
                                                     min_delta = 0, 
                                                     mode = 'auto', 
                                                     baseline = None, 
                                                     restore_best_weights = True)

gridUpdateCallback = getGridCallback( x_train, batch_size = 256)

trainingCallbacks = [ learningRateCallback, stoppingCallback, gridUpdateCallback]

model = createArchitecturePBKAN( inputSize = inputSize, outputSize = outputSize, layerSize = 24)
model.summary()

trainingHistory = model.fit( x = x_train,
                             y = y_train,
                             validation_data = ( x_valid, y_valid),
                             epochs = epochs,
                             verbose = 1,
                             batch_size = batchSize,
                             shuffle = True,
                             callbacks = trainingCallbacks)

modelSavePath = './kan-models/fiber-model-' + timeStamp
historySavePath = './kan-models/fiber-model-history-' + timeStamp + ".json"

saveModelWeights( modelSavePath, model)
saveDictionary( historySavePath, trainingHistory.history)
