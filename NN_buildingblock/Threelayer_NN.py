#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 16:05:15 2018

@author: xuping
"""
import numpy as np
import matplotlib.pyplot as plt
from OneNN import load_data

def sigmoid(x):
    s=1/(1+np.exp(-x))
    return s

def relu(x):
    s=np.maximum(0,x)
    return s

def model(X,Y,learning_rate, num_iterations, lambd, keep_prob, print_cost=True):
    grads={}
    costs=[]               # keep track of cost
    m=X.shape[1]           # number of training examples
    layers_dims=[X.shape[0], 10, 15, 6]
    
    parameters=initialize_parameters(layers_dims)
    
    for i in range(0,num_iterations):
        if keep_prob == 1:
            a3, cache = forward_prop(X,parameters)
        elif keep_prob < 1:
            a3, cache = forward_prop_with_dropout(X,parameters,keep_prob)
            
        if lambd == 0:
            cost = compute_cost(a3, Y)
        else:
            cost = compute_cost_with_regularization(a3, Y, parameters, lambd)
            
        assert(lambd==0 or keep_prob==1)
        
        if lambd ==0 and keep_prob ==1:
            grads=backprop(X,Y,cache)
        elif lambd != 0:
            grads=backprop_with_regularization(X,Y,cache,lambd)
        elif keep_prob <1:
            grads=backprop_with_dropout(X,Y,cache, keep_prob)
            
        parameters=update_parameters(parameters, grads, learning_rate)
        
        if print_cost and i % 1000 == 0:
            print("cost after iteration {}: {}".format(i,cost))
        if print_cost and i % 1000 == 0:
            costs.append(cost)
    
    #plt.plot(costs)
    #plt.ylabel('cost')
    #plt.xlabel('iterations(x1,000')
    #plt.title("learning rate=" + str(learning_rate))
    #plt.show()
    
    return parameters, costs


def forward_prop(X, parameters):
    W1=parameters["W1"]
    b1=parameters["b1"]
    W2=parameters["W2"]
    b2=parameters["b2"]
    W3=parameters["W3"]
    b3=parameters["b3"]
    
    Z1=np.dot(W1, X) + b1
    A1=relu(Z1)
    Z2=np.dot(W2, A1) + b2
    A2=relu(Z2)
    Z3=np.dot(W3, A2) + b3
    A3=sigmoid(Z3)
    
    cache=(Z1,A1,W1,b1,Z2,A2,W2,b2,Z3,A3,W3,b3)
    return A3, cache

def backprop(X,Y,cache):
    m=X.shape[1]
    (Z1,A1,W1,b1,Z2,A2,W2,b2,Z3,A3,W3,b3)=cache
    dZ3 = 2*(A3-Y)*sigmoid(Z3)*(1-sigmoid(Z3))
    dW3 = 1./m * np.dot(dZ3, A2.T)
    db3 = 1./m * np.dot(dZ3, axis=1, keepdims=True)
    
    dA2 = np.dot(W3.T,dZ3)
    dZ2 = np.multiply(dA2,np.int64(A2>0))
    dW2 = 1./m * np.dot(dZ2, A1.T)
    db2 = 1./m * np.sum(dZ2, axis=1, keepdims=True)
    
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1>0))
    dW1 = 1./m * np.dot(dZ1, X.T)
    db1 = 1./m * np.sum(dZ1, axis=1, keepdims=True)
    
    gradients = {"dZ3":dZ3, "dW3":dW3, "db3":db3, "dA2":dA2,
                 "dZ2":dZ2, "dW2":dW2, "db2":db2, "dA1":dA1,
                 "dZ1":dZ1, "dW1":dW1, "db1":db1}
    return gradients

def backprop_with_regularization(X,Y,cache,lambd):
    m=X.shape[1]
    (Z1,A1,W1,b1,Z2,A2,W2,b2,Z3,A3,W3,b3)=cache
    
    dZ3 = 2*(A3-Y)*sigmoid(Z3)*(1-sigmoid(Z3))
    dW3 = 1./m * np.dot(dZ3, A2.T) + lambd/m*W3
    db3 = 1./m * np.sum(dZ3, axis=1, keepdims=True)
    
    dA2 = np.dot(W3.T,dZ3)
    dZ2 = np.multiply(dA2,np.int64(A2>0))
    dW2 = 1./m * np.dot(dZ2, A1.T) + lambd/m*W2
    db2 = 1./m * np.sum(dZ2, axis=1, keepdims=True)
    
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1>0))
    dW1 = 1./m * np.dot(dZ1, X.T) + lambd/m*W1
    db1 = 1./m * np.sum(dZ1, axis=1, keepdims=True)
    
    gradients = {"dZ3":dZ3, "dW3":dW3, "db3":db3, "dA2":dA2,
                 "dZ2":dZ2, "dW2":dW2, "db2":db2, "dA1":dA1,
                 "dZ1":dZ1, "dW1":dW1, "db1":db1}
    return gradients

def initialize_parameters(layer_dims):
    np.random.seed(1)
    parameters={}
    L=len(layer_dims)
    
    for l in range(1,L):
        parameters["W"+str(l)]=np.random.randn(layer_dims[l],layer_dims[l-1]) / np.sqrt(layer_dims[l-1])
        parameters["b"+str(l)]=np.zeros((layer_dims[l],1))
        
        #assert(parameters["W"+str(l)].shape==layer_dims[l], layer_dims[l-1])
        #assert(parameters["b"+str(l)].shape==layer_dims[l], 1)
        
    return parameters
    
def update_parameters(parameters, grads, learning_rate):
    n = len(parameters)//2
    for k in range(n):
        parameters["W"+str(k+1)] = parameters["W"+str(k+1)]-learning_rate*grads["dW"+str(k+1)]
        parameters["b"+str(k+1)] = parameters["b"+str(k+1)]-learning_rate*grads["db"+str(k+1)]
        
    return parameters

def compute_cost(a3, Y):
    cost = np.sum(np.sum(np.square(a3-Y)))
    return cost

def compute_cost_with_regularization(A3, Y, parameters, lambd):
    
    m=Y.shape[1]
    W1=parameters["W1"]
    W2=parameters["W2"]
    W3=parameters["W3"]
    
    least_square_cost = compute_cost(A3, Y)
    L2_regularization_cost = lambd/(2*m)*(np.sum(np.sum(W1*W1))+np.sum(np.sum(W2*W2))+np.sum(np.sum(W3*W3)))
    
    cost = 1./m*least_square_cost + L2_regularization_cost
    return cost

def predic(X,y,parameters):
    m=X.shape[1]
    #p=np.zeros((1,m),dtype=np.int)
    a3,caches = forward_prop(X,parameters)
    
    #for i in range(0, a3.shape[1]):
    return a3

if __name__ == "__main__":
    X, Y5, Y6, Y10 = load_data()
    X5 = X[:5, :]
    X6 = X[:6, :]
    X10 = X[:10, :]
    
    num_iterations = 10000
    lambd = 0.5
    learning_rate = 0.5
    keep_prob = 1
    
    parameters,costs=model(X6,Y6,learning_rate, num_iterations, lambd, keep_prob, print_cost=True)
    
