import h5py
import numpy as np
import pandas as pd
import json

def loadData( fileName, maximumDataSets = 10000000):

    print( fileName )

    data = h5py.File( fileName, 'r')

    xData = data.get("x")
    xData = np.array( xData )

    yData = data.get( "y" )
    yData = np.array( yData )

    print( data )
    print( xData.shape )
    print( yData.shape )

    nPointsX = xData.shape[0]
    nPointsY = yData.shape[0]

    nDataSets = yData.shape[1]

    if( nDataSets > maximumDataSets ):
        nDataSets = maximumDataSets

    xDims = ( nDataSets, nPointsX, 1)
    yDims = ( nDataSets, nPointsY)

    X = np.empty( xDims )
    y = np.empty( yDims )

    for ii in range( nDataSets ):

        X[ ii, :, 0] = np.array( xData[ :, ii] ) - 0.5
        y[ ii, :] = np.array( yData[ :, ii] )

    #X[ :, 1:3] = np.log( X[ :, 1:3] )
    #X[ :, 16:18] = np.log( X[ :, 16:18] )
    #X[ :, 3] = X[ :, 3] / 45

    y[ :, 0:2] = np.log( y[ :, 0:2] )
    y[ :, 4] = np.log( y[ :, 4] )

    #X = X[ :, 1:, :]

    return X, y

def saveModelWeights( savePath, model):

    weights = model.get_weights()
    nWeights = len( weights )

    file = h5py.File( savePath, 'w')

    for ii in range( nWeights ):

        fieldII = 'weight' + str( ii )
        weightII = weights[ii]

        file.create_dataset( fieldII, data = weightII)

    file.close()


def saveDictionary( savePath, d):

    dictionaryDataFrame = pd.DataFrame( d ) 

    with open( savePath, mode = 'w') as file:
        dictionaryDataFrame.to_json( file )