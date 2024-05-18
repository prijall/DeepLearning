import numpy as np

class FeedForwardNetwork_Vectorised:

    def __init__(self):
#@ Sets the random seed to ensure reproducibility. 
        np.random.seed(0) 
#@ Initializes weights (w1) for connections from input to hidden layer with a shape of (2, 2), indicating 2 input features and 2 hidden units.
        self.w1=np.random.randn(2,2) 
#@ Initializes weights (w2) for connections from hidden to output layer with a shape of (2, 1), indicating 2 hidden units and 1 output unit.                                   
        self.w2=np.random.randn(2,1)
#@ Initializes biases (b1) for the hidden layer with a shape of (1, 2), indicating 2 biases for 2 hidden units.
        self.b1=np.zeros((1, 2))
#@ Initializes biases (b2) for the output layer with a shape of (1, 1), indicating 1 bias for the output unit.
        self.b2=np.zeros((1,1))

#@ Implements the sigmoid activation function, which maps any input value to a value between 0 and 1.
    def sigmoid(self, X):
        return 1/(1+ np.exp(-X))
    
    def forward_pass(self, X):
        """ Computes the forward pass through the neural network given an input X"""
#@ Computes the activations (a1) of the hidden layer by multiplying the input X with the weights w1, adding the biases b1, and then applying a linear transformation.
        self.a1=np.matmul(X, self.w1) + self.b1
#@ Applies the sigmoid activation function to the hidden layer activations, resulting in hidden layer outputs (h1). 
        self.h1=self.sigmoid(self.a1)
#@ Computes the activations (a2) of the output layer by multiplying the hidden layer outputs h1 with the weights w2, adding the biases b2, and then applying a linear transformation.
        self.a2=np.matmul(self.h1, self.w2) + self.b2
#@ Applies the sigmoid activation function to the output layer activations, resulting in the final output (h2).
        self.h1=self.sigmoid(self.a1)
#@  represents the output of the neural network.
        return self.h2

#@ Creates an instance of the FeedForwardNetwork_Vectorised class.  
ffn_v=FeedForwardNetwork_Vectorised()

#@ Calls the forward_pass method of the instance ffn_v with input X, where X should be a numpy array representing the input to the neural network.
ffn_v.forward_pass()