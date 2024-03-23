import numpy as np
from layer import Layer

class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)
        self.bias  = np.random.randn(output_size, 1)
    
    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias
    
    def backward(self, output_gradient, learning_rate):
        d_bias = output_gradient * learning_rate
        d_weights =  np.dot(output_gradient, self.input.T) * learning_rate
        
        self.weights -= d_weights
        self.bias -= d_bias
        
        return np.dot(self.weights.T, output_gradient)