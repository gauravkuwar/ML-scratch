import numpy as np
from scipy import signal
from layer import Layer

class Convolution(Layer):
    def __init__(self, input_shape, kernel_shape, kernel_depth):
        self.input_depth, input_height, input_width = input_shape
        self.input_shape = input_shape
        
        self.kernel_depth = kernel_depth
        kernel_height, kernel_width = kernel_shape
        self.kernel_shape = self.kernel_depth, self.input_depth, kernel_height, kernel_width
        
        output_shape = (kernel_depth, input_height - kernel_height + 1, input_width - kernel_width + 1)
        
        self.kernel = np.random.randn(*self.kernel_shape)
        self.biases = np.random.randn(*output_shape)
    
    def forward(self, input):
        self.input = input
        self.output = np.copy(self.biases)
        for i in range(self.kernel_depth): # for every kernel
            for j in range(self.input_depth): # for every input channel
                self.output[i] += signal.correlate2d(self.input[j], self.kernel[i, j], 'valid')
        return self.output
    
    def backward(self, output_gradient, learning_rate):
        kernel_grad = np.zeros(self.kernel_shape)
        input_grad = np.zeros(self.input_shape)
        
        for i in range(self.kernel_depth):
            for j in range(self.input_depth):
                kernel_grad[i, j] = signal.correlate2d(self.input[j], output_gradient[i], 'valid')
                input_grad[j] += signal.convolve2d(output_gradient[i], self.kernel[i, j], 'full')
                
        self.kernel -= kernel_grad * learning_rate
        self.biases -= output_gradient * learning_rate
        return input_grad