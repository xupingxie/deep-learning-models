#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Example of CNN on Tensorflow

@author: xuping
"""
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.framework import ops


def load_dataset():
    """
    load data from the sign image
    """
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

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    generate random minibatches from (X, Y)
    
    Input
    X -- input data, of shape (input size, number of examples) (m, Hi, Wi, Ci)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples) (m, n_y)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    m = X.shape[0]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]
    shuffled_Y = Y[permutation,:]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = int(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

def CNN_forward(X, parameters):
    """
    Implements the forward propagation for the CNN model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED
    
    Input
    X -- training dataset placeholder (input size, number of examples)
    parameters -- containing  "W1", "W2" filter info

    Output:
    Z3 -- the output of the last LINEAR unit
    """
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    
    #Conv layer, filter W1, stride = 1
    Z1 = tf.nn.conv2d(X, W1, strides=[1,1,1,1], padding='SAME')
    #relu layer
    A1 = tf.nn.relu(Z1)
    # max-pooling layer, filter=8, stride=8
    P1 = tf.nn.max_pool(A1, ksize=[1,8,8,1], strides=[1,8,8,1], padding='SAME')
    # conv layer, filter W2, stride = 1
    Z2 = tf.nn.conv2d(P1, W2, strides=[1,1,1,1], padding='SAME')
    A2 = tf.nn.relu(Z2)
    # max pooling layer
    P2 = tf.nn.max_pool(A2, ksize=[1,4,4,1], strides=[1,4,4,1], padding='SAME')
    # flatten
    P2 = tf.contrib.layers.flatten(P2)
    Z3 = tf.contrib.layers.fully_connected(P2, 6, activation_fn = None)
    
    return Z3

def CNN_model(X_train, Y_train, X_test, Y_test, learning_rate = 0.005, num_epochs = 100,
              minibatch_size = 64, print_cost = True):
    """
    This is a three-layer ConvNet model in Tensorflow:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED
    
    Input:
    X_train -- training set, of shape (None, 64, 64, 3)
    Y_train -- test set, of shape (None, n_y = 6)
    X_test -- training set, of shape (None, 64, 64, 3)
    Y_test -- test set, of shape (None, n_y = 6)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 5 epochs
    
    Output:
    train_accuracy -- real number, accuracy on the train set (X_train)
    test_accuracy -- real number, testing accuracy on the test set (X_test)
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    #rerun the model without overwriting tf variables
    ops.reset_default_graph()
    tf.set_random_seed(1)  # tensorflow seed keep consistent
    seed = 3   # numpy seed
    (m, H, W, C) = X_train.shape
    ny = Y_train.shape[1]
    costs = []
    
    # create tf placeholder
    X = tf.placeholder("float", [None, H, W, C])
    Y = tf.placeholder(tf.float32, [None, ny])
    
    # initialize parameters
    W1 = tf.get_variable("W1", [4,4,3,8], 
                         initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W2 = tf.get_variable("W2", [2,2,8,16],
                         initializer=tf.contrib.layers.xavier_initializer(seed=0))
    para = {"W1": W1, "W2": W2}
    
    # doing forward prop
    Z3 = CNN_forward(X, para)
    
    # compute cost
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3, labels=Y))
    
    #tensorflow optimizer 
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    
    #Initialize all variables globally
    init = tf.global_variables_initializer()
    
    # start session to compute the tensorgraph
    with tf.Session() as sess:
        sess.run(init)  # run initialization
        
        for epoch in range(num_epochs):
            
            minibatch_cost = 0.
            # number of minibatches of size minibatch_size
            num_minibatches = int(m / minibatch_size)
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
            
            for minibatch in minibatches:
                
                # select minibatch
                (minibatch_X, minibatch_Y) = minibatch
                
                # run the session
                _, temp_cost = sess.run([optimizer, cost], feed_dict={
                        X:minibatch_X, Y:minibatch_Y})
                
                minibatch_cost += temp_cost / num_minibatches
            
            #print cost
            if print_cost and epoch % 5 == 0:
                print("cost after epoch % i: %f" %(epoch, minibatch_cost))
            if print_cost and epoch % 1 == 0:
                costs.append(minibatch_cost)
                
        plt.plot(np.squeeze(cost))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate=" + str(learning_rate))
        plt.show()
        
        #calculate predictions.
        predict_op = tf.argmax(Z3, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
        
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(accuracy)
        train_accuracy = accuracy.eval({X:X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X:X_test, Y:Y_test})
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)
        
        return train_accuracy, test_accuracy, para

if __name__ == "__main__":
    #load data
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
    index = 6
    plt.imshow(X_train_orig[index])
    print ("y = " + str(np.squeeze(Y_train_orig[:, index])))
    
    X_train = X_train_orig/255.
    X_test = X_test_orig/255.
    Y_train = convert_to_one_hot(Y_train_orig, 6).T
    Y_test = convert_to_one_hot(Y_test_orig, 6).T
    print ("number of training examples = " + str(X_train.shape[0]))
    print ("number of test examples = " + str(X_test.shape[0]))
    print ("X_train shape: " + str(X_train.shape))
    print ("Y_train shape: " + str(Y_train.shape))
    print ("X_test shape: " + str(X_test.shape))
    print ("Y_test shape: " + str(Y_test.shape))
    conv_layers = {}
    
    _, _, parameters = CNN_model(X_train, Y_train, X_test, Y_test)
    
    
    

                

                    