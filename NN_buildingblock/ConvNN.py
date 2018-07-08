#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This script contains basic functions for Conv Neural Nets.
foward conv and pooling
backward conv and pooling

@author: xuping
"""
import numpy as np
import h5py
import matplotlib.pyplot as plt


def Conv_forward(A_prev, W, b, para):
    '''
    This is the forward propagation for a convolution layer
    
    Input: output from previous layer A_prev (m, H_prev, W_prev, C_prev)
    W  --- weights, (f,f, C_prev, C)
    b  --- bias,
    para --- contains "stride" and "pad"
    
    return the conv output Z(m, H, W, C), cache for backpropagation
    '''
    (m, H_prev, W_prev, C_prev) = A_prev.shape
    (f, f, C_prev, C) = W.shape
    stride = para["stride"]
    pad = ["pad"]
    
    H = int((H_prev - f + 2 * pad) / stride + 1)
    W = int((W_prev - f + 2 * pad) / stride + 1)
    Z = np.zeros((m, H, W, C))
    
    # padding the input
    A_prev_pad = np.pad(A_prev, ((0,0), (pad,pad), (pad,pad), (0,0)), 'constant', constant_value=(0,0))
    
    # loop all dimension
    for i in range(m):
        a_prev_pad = A_prev_pad[i,:,:,:]   # extract the i-th training example
        for h in range(H):
            for w in range(W):
                for c in range(C):
                    hstart = stride * h
                    hend = hstart + f
                    wstart = stride * w
                    wend = wstart + f
                    # extract the slice for Conv
                    a_slice = a_prev_pad[hstart:hend, wstart:wend, :]
                    # Conv step
                    Z[i,h,w,c] = np.sum(a_slice * W[:,:,:,c]) + b[:,:,:,c] 
                    
    #end for loop
    assert(Z.shape == (m, H, W, C))
    # save in cache for backprop
    cache = (A_prev, W, b, para)
    
    return Z, cache

def Pool_forward(A_prev, para, mode="max"):
    '''
    forward progation of pooling layer
    Input: A_prev(m, H_prev, W_prev, C_prev)
    para -- parameters
    mode -- max pooling or average
    
    output: pooling output layer A(m, H, W, C)
    '''
    (m, H_prev, W_prev, C_prev) = A_prev.shape
    f = para["f"]
    stride = para["stride"]
    
    H = int((H_prev - f) / stride + 1)
    W = int((W_prev - f) / stride + 1)
    C = C_prev
    # initialize output A
    A = np.zeros((m, H, W, C))
    # loop each dimension
    for i in range(m):
        for h in range(H):
            for w in range(W):
                for c in range(C):
                    hstart = stride * h
                    hend = hstart + f
                    wstart = stride * w
                    wend = wstart + f
                    # extract the slice from A_prev
                    a_slice = A_prev[i, hstart:hend, wstart:wend, c]
                    
                    if mode == "max":
                        A[i,h,w,c] = np.max(a_slice)
                    elif mode == "average":
                        A[i,h,w,c] = np.mean(a_slice)
    # end for loop
    assert(A.shape == (m, H, W, C))
    cache = (A_prev, para)
    
    return A, cache

def Conv_backward(dZ, cache):
    '''
    the backward propgation of Conv Layer
    
    Input: dZ -- gradient of the cost wrt the OUTPUT of Conv Layer Z (m, H, W, C)
    cache -- stored data from forward prop
    
    Output: dA -- gradient of the cost wrt INPUT of Conv layer A_prev (m, H_prev, W_prev, C_prev)
    dW -- gradient wrt weights of the Conv layer W(f, f, C_prev, C)
    db -- gradient wrt biases b(1,1,1,C)
    '''
    # get all the dimensions from previous data
    (A_prev, W, b, para) = cache
    (m, H_prev, W_prev, C_prev) = A_prev.shape
    (f, f, C_prev, C) = W.shape
    stride = para["stride"]
    pad = para["pad"]
    (m, H, W, C) = dZ.shape
    #intialize all the gradients
    dA_prev = np.zeros((m, H_prev, W_prev, C_prev))
    dW = np.zeros((f, f, C_prev, C))
    db = np.zeros(b.shape)
    
    #padding the data
    A_prev_pad = np.pad(A_prev, ((0,0), (pad,pad), (pad,pad),(0,0)),'constant', constant_value=(0,0))
    dA_prev_pad = np.pad(dA_prev, ((0,0), (pad,pad), (pad,pad),(0,0)),'constant', constant_value=(0,0))
    
    #loop all the dimensions
    for i in range(m):
        a_prev = A_prev_pad[i,:,:,:]
        da_prev = dA_prev_pad[i,:,:,:]
        
        for h in range(H):
            for w in range(W):
                for c in range(C):
                    # define the corner of the slice
                    hstart = stride * h
                    hend = hstart + f
                    wstart = stride * w
                    wend = wstart + f
                    
                    #extract slice
                    a_slice = a_prev[hstart:hend, wstart:wend, :]
                    
                    # compute the derivate
                    da_prev[hstart:hend, wstart:wend,:] += W[:,:,:,c]*dZ[i,h,w,c]
                    dW[:,:,:,c] += a_slice*dZ[i,h,w,c]
                    db[:,:,:,c] += dZ[i,h,w,c]
        
        #remove pad from the local derivative slice
        dA_prev[i,:,:,:] = da_prev[pad:-pad, pad:-pad, :]
    #end for loop
    assert(dA_prev.shape == (m, H, W, C))
    
    return dA_prev, dW, db

def Pooling_backward(dA, cache, mode="max"):
    """
    Find gradients through backward prop of the pooling layer 
    
    Input: dA -- gradients wrt OUTPUT of the pooling layer 
    cache -- stored output data from forward prop
    mode -- max pooling or average
    
    Output: dA_prev -- the gradient wrt the INPUT of the pooling layer
    """
    (A_prev, para) = cache
    
    stride = para["stride"]
    f = para["f"]
    m, H_prev, W_prev, C_prev = A_prev.shape
    m, H, W, C = dA.shape
    
    #Initialize dA_prev with zeros
    dA_prev = np.zeros((m, H_prev, W_prev, C_prev))
    
    #loop all the dimensions
    for i in range(m):
        # extract the training exmaple from A_prev
        a_prev = A_prev[i,:,:,:]
        
        for h in range(H):
            for w in range(W):
                for c in range(C):
                    # define the corner of the slice
                    hstart = stride * h
                    hend = hstart + f
                    wstart = stride * w
                    wend = wstart + f
                    
                    # compute the backprop 
                    if mode == "max":
                        # extract the slice
                        a_slice = a_prev[hstart:hend, wstart:wend, c]
                        # create mask for the slice matrix
                        mask = (a_slice == np.max(a_slice))
                        # compute derivative
                        dA_prev[i, hstart:hend, wstart:wend, c] += mask*dA[i,h,w,c]
                        
                    elif mode == "average":
                        # get the value
                        da = dA[i,h,w,c]
                        
                        # compute the derivative
                        dA_prev[i, hstart:hend, wstart:wend, c] += da/(f+f)*np.ones((f,f))
    # end loop
    assert(dA_prev.shape == A_prev.shape)
    
    return dA_prev

    