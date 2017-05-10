#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import math,random

def sigmoid(x): 
    return 1/(1 + np.exp(-2*x))      # activation function
    
def sigmoid_d(x): 
    return x * (1 - x)             # derivative of sigmoid


class Perceptron:
    def __init__(self,threshold=0.5, learning_rate=0.2, error_margin=0.01,add_bias=True, **kwargs):
        ########################
        #  coming soon!
        ########################
        
    def _add_bias(self,inputs):
        """Add an additional constant 1 bias to the inputs to model the threshold"""
        return [(inp+[1],t) for inp,t in inputs]
        
    def train(self,inputs):
        """update the weights of the modeled neuron iteratively based on some input data in the form (input_vector,desired_output)"""
        ######################
        # coming soon!
        ######################
    
    def predict(self,data):
        """predict instances based on the trained model"""
        #check if we predict a list of vectors or just one vector
        if type(data)==list and type(data[0])==list:
            output=[]
            if self.bias:
                data=self._add_bias(data)
            for input_vector in test_points:
                pred=np.dot(input_vector, self.weights) > self.threshold
                output.append(pred)
        else:
            output = np.dot(data, self.weights) > self.threshold
        return output
            
                
    def test(self,test_data):
        """test the learned accuracy given on a test set of data"""
        if self.bias:
            test_data=self._add_bias(test_data)
        correct=0
        for input_vector, desired_output in test_data:
            pred=self.predict(input_vector)
            if pred==desired_output:
                correct+=1
        print "accuracy: ", float(correct)/float(len(test_data))
        return float(correct)/float(len(test_data))
            
    
if __name__=="__main__":
    training_set_nand = [([0,0],1),([0,1],1),([1,0],1),([1,1],0)]
    #xor with a third dimension x1*x2
    training_set_xor = [([0,0,0],0),([0,1,0],1),([1,0,0],1),([1,1,1],0)]
    p=Perceptron(verbose=True,max_iterations=100,activation_function=sigmoid)
    p.train(training_set_xor)
    p.test(training_set_xor)
