import keras
from keras.layers import Activation, BatchNormalization, Conv1D, Dense, Dropout, Flatten, Input, Normalization
from keras.models import Sequential
from tensorflow.keras.regularizers import l2

from keras.initializers import RandomNormal

import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.keras.optimizers import Adam

from tfkan.layers import DenseKAN, Conv1DKAN
from keras import regularizers
import numpy as np

tfd = tfp.distributions
tfpl = tfp.layers

import tfkan as tft

optimizer = Adam( learning_rate = 0.001 )

# define update grid callback
class UpdateGridCallback(tf.keras.callbacks.Callback):
    def __init__(self, training_data, batch_size: int = 128):
        super(UpdateGridCallback, self).__init__()
        self.training_data = tf.constant(training_data[:batch_size], tf.float32)
        
    def on_epoch_begin(self, epoch, logs=None):
        """
        update grid before new epoch begins
        """
        #print("Updating KAN grids.")
        x_batch = self.training_data

        #print( self.training_data[:,0,0] )

        if epoch > -1:
            for layer in self.model.layers:
                if hasattr(layer, 'update_grid_from_samples'):

                # Extract the inbound layer of the KAN layer
                
                    layerInput = layer._inbound_nodes[0].inbound_layers

                    intermediate = keras.Model( inputs = self.model.inputs,
                                                outputs = layerInput.output)
                    
                    kanInputTensor = intermediate( x_batch )
                    #print( )
                    #print( np.min( kanInputTensor ), np.max( kanInputTensor ))
                    #print( np.min( layer( kanInputTensor ) ), np.max( layer( kanInputTensor ) ))
                    #print( )
                

                    try:
                        layer.update_grid_from_samples( kanInputTensor )
                        #print("Grid updated.")
                    except:
                        pass
                        #print("Skipping grid update for one KAN layer.")
                        

def getGridCallback( x, batch_size):
    return UpdateGridCallback( x, batch_size = batch_size)


def createArchitecturePBKAN( inputSize, outputSize, layerSize = 1):

    inputShape = ( inputSize, 1)

    negativeLogLikelihood = lambda y, rv_y: -rv_y.log_prob( y )

    elasticModuli = lambda input: tfd.Normal( loc = 20 * tf.nn.sigmoid( input[..., 0:2] ) - 10,
                                                 scale = 10 * tf.nn.sigmoid( input[..., 2:4] ) + 1e-6 )

    poissonRatios = lambda input: tfd.Normal( loc   = 6 * tf.nn.sigmoid( input[..., 0:2] ) - 3,
                                              scale = 10 * tf.nn.sigmoid( input[..., 2:4] ) + 1e-6 )
    
    elasticModulus = lambda input: tfd.Normal( loc = 20 * tf.nn.sigmoid( input[..., 0:1] ) - 10,
                                                  scale = 10 * tf.nn.sigmoid( input[..., 1:2] ) + 1e-6 )

    elasticModuliLayer = tfp.layers.DistributionLambda( elasticModuli )
    poissonRatioLayer = tfp.layers.DistributionLambda( poissonRatios )
    elasticModulusLayer = tfp.layers.DistributionLambda( elasticModulus )

    inputs = Input( shape = inputShape )
    
    x = BatchNormalization(  axis = -1,
                                    momentum = 0.99,
                                    epsilon = 0.001,
                                    center = True,
                                    scale = True,
                                    beta_initializer = 'zeros',
                                    gamma_initializer = 'ones',
                                    moving_mean_initializer = 'zeros',
                                    moving_variance_initializer = 'ones',
                                    beta_regularizer = None,
                                    gamma_regularizer = None,
                                    beta_constraint = None,
                                    gamma_constraint = None,
                                    input_shape = inputShape
                                 )( inputs )


    xBayesian = tfp.layers.DenseFlipout( layerSize )( 0 * inputs )
    x = tft.layers.DenseKAN( layerSize )( inputs )

    x = keras.layers.concatenate([ xBayesian / np.sqrt( layerSize ), x])
    x = Flatten()( x )
    x = tft.layers.DenseKAN( 5 * layerSize )( x )
    x = Flatten()( x )
    x = tft.layers.DenseKAN( 5 * layerSize )( x )
    x = Flatten()( x )
    x = tft.layers.DenseKAN( outputSize + outputSize )( x )

    E = elasticModuliLayer( x[ ..., 0:4] )
    nu = poissonRatioLayer( x[ ..., 4:8] )
    mu = elasticModulusLayer( x[ ..., 8:10] )

    model = keras.Model( inputs = inputs, outputs = [ E, nu, mu])
    model.compile( loss = negativeLogLikelihood, optimizer = optimizer)

    return model


def createArchitecturePBMLP( inputSize, outputSize, layerSize = 1):

    inputShape = ( inputSize, 1)

    negativeLogLikelihood = lambda y, rv_y: -rv_y.log_prob( y )

    elasticModuli = lambda input: tfd.Normal( loc = 20 * tf.nn.sigmoid( input[..., 0:2] ) - 10,
                                                 scale = 10 * tf.nn.sigmoid( input[..., 2:4] ) + 1e-6 )

    poissonRatios = lambda input: tfd.Normal( loc   = 6 * tf.nn.sigmoid( input[..., 0:2] ) - 3,
                                              scale = 10 * tf.nn.sigmoid( input[..., 2:4] ) + 1e-6 )
    
    elasticModulus = lambda input: tfd.Normal( loc = 20 * tf.nn.sigmoid( input[..., 0:1] ) - 10,
                                                  scale = 10 * tf.nn.sigmoid( input[..., 1:2] ) + 1e-6 )

    elasticModuliLayer = tfp.layers.DistributionLambda( elasticModuli )
    poissonRatioLayer = tfp.layers.DistributionLambda( poissonRatios )
    elasticModulusLayer = tfp.layers.DistributionLambda( elasticModulus )

    inputs = Input( shape = inputShape )

    #x = Normalization()( inputs)
    initializer = RandomNormal( mean=0, stddev=1)

    xBayesian = tfp.layers.DenseFlipout( layerSize )( 0 * inputs )
    x = Dense( 1 * layerSize, activation = "relu")( inputs )

    #x = keras.layers.concatenate([ xBayesian / np.sqrt( layerSize ), x ])
    x = keras.layers.concatenate([ xBayesian / layerSize, x])
    x = Flatten()( x )
    x = Dense( 5 * layerSize, activation = "relu")( x )
    x = Flatten()( x )
    x = Dense( 5 * layerSize, activation = "relu")( x )
    x = Flatten()( x )
    x = Dense( outputSize + outputSize )( x )


    E = elasticModuliLayer( x[ ..., 0:4] )
    nu = poissonRatioLayer( x[ ..., 4:8] )
    mu = elasticModulusLayer( x[ ..., 8:10] )

    model = keras.Model( inputs = inputs, outputs = [ E, nu, mu])
    model.compile( loss = negativeLogLikelihood, optimizer = optimizer)

    return model


def createArchitectureDropoutMLP( inputSize, outputSize, layerSize = 1):

    inputShape = ( inputSize, 1)

    negativeLogLikelihood = lambda y, rv_y: -rv_y.log_prob( y )

    elasticModuli = lambda input: tfd.Normal( loc = 20 * tf.nn.sigmoid( input[..., 0:2] ) - 10,
                                                 scale = 10 * tf.nn.sigmoid( input[..., 2:4] ) + 1e-6 )

    poissonRatios = lambda input: tfd.Normal( loc   = 6 * tf.nn.sigmoid( input[..., 0:2] ) - 3,
                                              scale = 10 * tf.nn.sigmoid( input[..., 2:4] ) + 1e-6 )
    
    elasticModulus = lambda input: tfd.Normal( loc = 20 * tf.nn.sigmoid( input[..., 0:1] ) - 10,
                                                  scale = 10 * tf.nn.sigmoid( input[..., 1:2] ) + 1e-6 )

    elasticModuliLayer = tfp.layers.DistributionLambda( elasticModuli )
    poissonRatioLayer = tfp.layers.DistributionLambda( poissonRatios )
    elasticModulusLayer = tfp.layers.DistributionLambda( elasticModulus )

    inputs = Input( shape = inputShape )

    #x = Normalization()( inputs)
    initializer = RandomNormal( mean=0, stddev=1)

    xBayesian = tfp.layers.DenseFlipout( layerSize )( 0 * inputs )
    x = Dense( 2 * layerSize, activation = "relu", kernel_regularizer=l2(0.0))( inputs )

    #x = keras.layers.concatenate([ xBayesian / np.sqrt( layerSize ), x ])
    #x = keras.layers.concatenate([ xBayesian / layerSize, x])
    x = Flatten()( x )
    #x = Dropout( 0.0 )( x )
    x = Dense( 5 * layerSize, activation = "relu", kernel_regularizer=l2(0.0))( x )
    x = Flatten()( x )
    x = Dropout( 0.2 )( x )
    x = Dense( 5 * layerSize, activation = "relu", kernel_regularizer=l2(0.0) )( x )
    x = Flatten()( x )
    x = Dense( outputSize + outputSize )( x )


    E = elasticModuliLayer( x[ ..., 0:4] )
    nu = poissonRatioLayer( x[ ..., 4:8] )
    mu = elasticModulusLayer( x[ ..., 8:10] )

    model = keras.Model( inputs = inputs, outputs = [ E, nu, mu])
    model.compile( loss = negativeLogLikelihood, optimizer = optimizer)

    return model


def getNumberOfTrainableParameters( model ):
    return keras.utils.layer_utils.count_params( model.trainable_weights )