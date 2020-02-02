# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 22:33:10 2020

@author: Ritesh
"""
import numpy as np
import matplotlib.pyplot as plt

def initialize_parameters(layers):
    params = {}
    for i in range(len(layers)):
        if(i == len(layers)-1):
            break
        params['Weights'+str(i+1)] = np.random.randn(layers[i+1],layers[i]) / np.sqrt(layers[i])  # xavier intialization
        params['b'+str(i+1)] = np.zeros((layers[i+1],1))
    return params
# sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))
# relu function
def relu(x):
    return np.maximum(0,x)
def drelu(dA,Z):
    dA[Z<=0] = 0
    return dA
# forward pass for neural network
def forward(X,params):
    cache = {}
    len_params = int(len(params)/2)
    for i in range(len_params):
        if i ==0:
            Z = np.dot(params['Weights'+str(i+1)],X) + params['b'+str(i+1)]
        else:
            Z = np.dot(params['Weights'+str(i+1)],cache['A'+str(i)]) + params['b'+str(i+1)]
        if i < len_params-1:
            A = relu(Z)
        else:
            A = sigmoid(Z)
        cache['Z'+str(i+1)] = Z
        cache['A'+str(i+1)] = A
    return cache

def backward(X,Y,cache,params):
    grads = {}
    cost = -np.sum(Y*np.log(cache['A'+str(int(len(cache)/2))])+(1-Y)*np.log(1-cache['A'+str(int(len(cache)/2))]))/X.shape[1]
    for i in range(int(len(cache)/2),0,-1):
        if i == int(len(cache))/2:
            dZ = (cache['A'+str(i)] - Y)
            dW = np.dot(dZ,cache['A'+str(i-1)].T) / X.shape[1]
            db = np.sum(dZ,axis=1,keepdims=True) / X.shape[1]
        else:
            dZ = drelu(np.dot(params['Weights'+str(i+1)].T,grads['dZ'+str(i+1)]),cache['Z'+str(i)])
            if i == 1:
                dW = np.dot(dZ,X.T) / X.shape[1]
            else:
                dW = np.dot(dZ,cache['A'+str(i-1)].T) / X.shape[1]
            db = np.sum(dZ,axis=1,keepdims=True) / X.shape[1]
        grads['dZ'+str(i)] = dZ
        grads['dW'+str(i)] = dW
        grads['db'+str(i)] = db
    return cost,grads

#updating the weights
def update_weights(grads,params,lr):
    for i in range(int(len(params)/2)):
        params['Weights'+str(i+1)] = params['Weights'+str(i+1)] - lr*grads['dW'+str(i+1)] 
        params['b'+str(i+1)] = params['b'+str(i+1)] - lr*grads['db'+str(i+1)]
    return params
           
def model(X,Y,params,lr,iterations=2500):
    costs = []
    for i in range(iterations):
        cache = forward(X,params)
        cost,grads = backward(X,Y,cache,params)
        params = update_weights(grads,params,lr)
        if (i%100 == 0):
            print(cost)
            costs.append(cost)
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(lr))
    plt.show()
    return cache,costs