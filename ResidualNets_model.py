#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Residual Networks model (ResNet50) for image classification
One of the assignments in deep learning on Coursera by Andrew Ng

This algorithm due to He et al. (2015)
Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun - Deep Residual Learning for Image Recognition (2015)

@author: xuping
"""
import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.initializers import glorot_uniform

import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)

def load_dataset():
    train_dataset = h5py.File('train_signs.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('test_signs.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

def identity_block(X, f, filters, stage, block):
    """
    compute the identity block
    
    Inputs:
    X -- input tensor of shape (m, H_prev, W_prev, C_prev)
    f -- the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- used to name the layers
    block -- string/character, used to name the layers, depending on their position in the network
    
    Outputs:
    X -- output of the identity block, tensor of shape (H, W, C)
    """
    #defning name basis
    conv_name_base = 'res' + str(stage) + block +'_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    #retrieve filters
    F1, F2, F3 = filters
    
    #save input value
    X_shortcut = X
    
    #first component
    X = Conv2D(filters = F1, kernel_size=(1,1), strides=(1,1), padding='valid',
               name = conv_name_base+'2a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base+'2a')(X)
    X = Activation('relu')(X)
    
    #second component
    X = Conv2D(filters = F2, kernel_size=(f,f), strides=(1,1), padding='SAME',
               name=conv_name_base+'2b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base+'2b')(X)
    X = Activation('relu')(X)
    
    #Third component
    X = Conv2D(filters = F3, kernel_size=(1,1), strides=(1,1), padding='valid',
               name=conv_name_base+'2c', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base+'2c')(X)
    
    #Add shortcut to main path 
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    return X

def convolutional_block(X, f, filters, stage, block, s = 2):
    """
    Implementation of the convolutional block
    
    Inputs:
    X -- input tensor of shape (m, H_prev, W_prev, C_prev)
    f -- specifying the shape of the middle CONV's window for the main path
    filters -- python list defining the number of filters in the CONV layers of the main path
    stage -- used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- specifying the stride to be used
    
    Outputs:
    X -- output of the convolutional block, tensor of shape (H, W, C)
    """
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    F1, F2, F3 = filters
    
    # save the input 
    X_shortcut = X
    
    # first component
    X = Conv2D(F1, (1,1), strides=(s,s), name=conv_name_base+'2a',
               padding='valid', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    #second component
    X = Conv2D(F2, (f,f), strides=(1,1), name = conv_name_base+'2b',
               padding='SAME', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name = bn_name_base+'2b')(X)
    X = Activation('relu')(X)
    
    #third    
    X = Conv2D(F3, (1,1), strides=(1,1), name = conv_name_base+'2c',
               padding='valid', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base+'2c')(X)
    
    #shortcut
    X_shortcut = Conv2D(F3, (1,1), strides=(s,s), name = conv_name_base+'1',
                        padding='valid', kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name = bn_name_base+'1')(X_shortcut)
    
    # add shortcut to main path
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    return X

def ResNet50(input_shape = (64,64,3), classes = 6):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """
    
    #define input tensor
    X_input = Input(input_shape)
    # zero padding
    X = ZeroPadding2D((3,3))(X_input)
    
    #stage 1
    X = Conv2D(64, (7,7), strides=(2,2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3,3), strides=(2,2))(X)
    
    #stage 2
    X = convolutional_block(X, f = 3, filters=[64,64,256], stage = 2, block='a', s=1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')
    
    #stage 3
    X = convolutional_block(X, f = 3, filters = [128, 128, 512], stage = 3, block = 'a', s = 2)
    X = identity_block(X, 3, [128, 128, 512], stage = 3, block = 'b')
    X = identity_block(X, 3, [128, 128, 512], stage = 3, block = 'c')
    X = identity_block(X, 3, [128, 128, 512], stage = 3, block = 'd')
    
    #stage 4
    X = convolutional_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block = 'a', s = 2)
    X = identity_block(X, 3, [256, 256, 1024], stage = 4, block = 'b')
    X = identity_block(X, 3, [256, 256, 1024], stage = 4, block = 'c')
    X = identity_block(X, 3, [256, 256, 1024], stage = 4, block = 'd')
    X = identity_block(X, 3, [256, 256, 1024], stage = 4, block = 'e')
    X = identity_block(X, 3, [256, 256, 1024], stage = 4, block = 'f')
    
    #stage 5
    X = convolutional_block(X, f = 3, filters =[512, 512, 2048], stage = 5, block='a', s = 2)
    X = identity_block(X, 3, [512, 512, 2048], stage = 5, block ='b')
    X = identity_block(X, 3, [512, 512, 2048], stage = 5, block ='c')
    
    #averge pooling
    X = AveragePooling2D(pool_size=(2,2), strides=None, padding='valid', name='avg_pool')(X)
    
    #output layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc'+str(classes), kernel_initializer=glorot_uniform(seed=0))(X)
    
    #create model
    model = Model(inputs = X_input, outputs = X, name='ResNet50')
    
    return model

if __name__ == "__main__":
    
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

    # Normalize image vectors
    X_train = X_train_orig/255.
    X_test = X_test_orig/255.

    # Convert training and test labels to one hot matrices
    Y_train = convert_to_one_hot(Y_train_orig, 6).T
    Y_test = convert_to_one_hot(Y_test_orig, 6).T

    print ("number of training examples = " + str(X_train.shape[0]))
    print ("number of test examples = " + str(X_test.shape[0]))
    print ("X_train shape: " + str(X_train.shape))
    print ("Y_train shape: " + str(Y_train.shape))
    print ("X_test shape: " + str(X_test.shape))
    print ("Y_test shape: " + str(Y_test.shape))
    
    # define our ResNet50 model
    model = ResNet50(input_shape=(64,64,3), classes = 6)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    model.fit(X_train, Y_train, epochs=2, batch_size=32)
    
    preds = model.evaluate(X_test, Y_test)
    print("Loss = " + str(preds[0]))
    print("Test accuracy = " + str(preds[1]))
    
    
    
