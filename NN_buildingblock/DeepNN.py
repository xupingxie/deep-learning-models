#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Building blocks of Deep NN

@author: xuping
"""

import numpy as np 
import h5py
import matplotlib.pyplot as plt

def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    return A

def sigmoid_backprop(dA, Z):
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    assert(dZ.shape == Z.shape)
    return dZ

def relu_backprop(dA, Z):
    
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    
    assert(dZ.shape == Z.shape)  
    return dZ

def DeepNN_initialize_parameters(layers):
    """
    This function initialize a L-layer deep NN parameters
    
    Input:
    layers -- python array containing the dimensions of each layer
    
    Output:
    parameters -- python dictionary including parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """
    np.random.seed(1)
    parameters = {}
    L = len(layers)   # number of layers in the NN
    
    # loop start from 1 layer since 0 layer is the input layer.
    for l in range(1, L):
        parameters["W"+str(l)] = np.random.randn(layers[l], layers[l-1])*0.01
        parameters["b"+str(l)] = np.zeros((layers[l],1))
        
        assert(parameters["W"+str(l)].shape == (layers[l], layers[l-1]))
        assert(parameters["b"+str(l)].shape == (layers[l], 1))
        
    return parameters

def forward_prop(A_prev, W, b, activation):
    """
    L layer NN forward propagation

    Inputs:
    A -- input data from previous layer: (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- activation to be used in this layer, "sigmoid" or "relu"
    Outputs:
    Z -- the input of the activation function 
    cache -- containing "A", "W" and "b" ; stored for computing the backward_prop.
    """
    
    Z = np.dot(W, A_prev) + b
    assert(Z.shape == (W.shape[0], A_prev.shape[1]))
    
    linear_cache = (A_prev, W, b)
    
    if activation == "sigmoid":
        A = 1/(1+np.exp(-Z))
        
    elif activation == "relu":
        A = np.maximum(0,Z)
        
    assert(A.shape == W.shape[0], A_prev.shape[1])
    cache = (linear_cache, Z)
    
    return A, cache
def DeepNN_forward(X, parameters):
    """
    forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID NN model
    
    Inputs:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()
    
    Outputs:
    AL -- last layer output (post-activation value)
    caches -- list of caches containing:
                every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
                the cache of linear_sigmoid_forward() (there is one, indexed L-1)
    """
    caches = []
    A = X
    L = len(parameters) // 2
    
    #[LINEAR -> RELU]*(L-1)
    for l in range(1, L):
        A_prev = A
        
        W = parameters["W" + str(l)]
        b = parameters["b" + str(l)]
        
        A, cache = forward_prop(A_prev, W, b, activation = "relu")
        caches.append(cache)
        
    # last layer sigmoid.
    W = parameters["W" + str(L)]
    b = parameters["b" + str(L)]
    
    AL, cache = forward_prop(A, W, b, activation = "sigmoid")
    caches.append(cache)
    
    assert(AL.shape == (1, X.shape[1]))
    
    return AL, caches

def compute_cost(AL, Y):
    """
    cross-entropy cost function
    
    Inputs:
    AL -- probability vector
    Y -- true "label" vector

    Outpus:
    cost -- cross-entropy cost
    """
    m = Y.shape[1]
    
    cost = -np.sum(Y*np.log(AL) + (1-Y)*np.log(1-AL)) / m
    cost = np.squeeze(cost)
    
    assert(cost.shape == ())
    return cost

def backward_prop(dZ, cache):
    """
    the backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    
    A_prev, W, b = cache
    m = A_prev.shape[1]
    
    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True)/m
    dA_prev = np.dot(W.T, dZ)
    
    assert(dA_prev.shape == A_prev.shape)
    assert(dW.shape == W.shape)
    assert(db.shape == b.shape)
    
    return dA_prev, dW, db
def backprop_activation(dA, cache, activation):
    
    linear_cache, Z = cache
    if activation == "sigmoid":
        dZ = sigmoid_backprop(dA, Z)
        dA_prev, dW, db = backward_prop(dZ, linear_cache)
        
    elif activation == "relu":
        dZ = relu_backprop(dA, Z)
        dA_prev, dW, db = backward_prop(dZ, linear_cache)
        
    return dA_prev, dW, db

 def DeepNN_backward(AL, Y, caches):
     """
    backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID NN model
    
    Inputs:
    AL -- probability vector, output of the DeepNN_forward propagation
    Y -- true "label" vector
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
    
    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ... 
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ... 
    """
    grads = {}
    L = len(caches) # number of layers
    m = AL.shape[1]  # number of training examples
    Y = Y.reshape(AL.shape) # Y is same shape as AL
    
    # initialize backprop
    dAL = -np.divide(Y, AL) + np.divide(1-Y, 1-AL)
    # Lth layer sigmoid->linear gradients, Input: AL, Y, caches
    current_cache = caches[L-1]
    grads["dA" + str(L)], grads["dW"+str(L)], grads["db"+str(L)] = backprop_activation(dA,
          current_cache, activation="sigmoid")
    
    #loop all the layer
    for l in reversed(range(L-1)):
        # lth layer: relu->linear gradients.
        #Input: dA(l+2), caches, output dA(l+1)
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = backprop_activation(
                grads["dA"+str(l+2))], current_cache, activation="relu")
        grads["dA"+str(l+1)] = dA_prev_temp
        grads["dW"+str(l+1)] = dW_temp
        grads["db"+str(l+1)] = db_temp
        
    return grads

def update_parameters(parameters, grads, learning_rate):
    """
    update parameters using gradient descent
    """
    L = len(parameters)//2
    #update parameters for each layer
    for l in range(L):
        parameters["W"+str(l+1)] = parameters["W"+str(l+1)]-learning_rate*grads["dW"+str(l+1)]
        parameters["b"+str(l+1)] = parameters["b"+str(l+1)]-learning_rate*grads["db"+str(l+1)]
    
    return parameters

    
    
    
    
    
    
        