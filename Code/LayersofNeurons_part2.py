import numpy as np
import nnfs
from nnfs.datasets import spiral_data



nnfs.init()

class Dense_layer:

    #@ Layers Initialization:
    def __init__(self, n_inputs, n_neurons, weight_regularizer_L1=0, 
                 weight_regularizer_L2=0, bias_regularizer_L1=0, bias_regularizer_L2=0):
        #@ Initializing weights and bias:
        self.weights=0.01*np.random.randn(n_inputs, n_neurons)
        self.biases=np.zeros((1, n_neurons))

    #@ Setting regularization strength
        self.weight_regularizer_L1=weight_regularizer_L1
        self.weight_regularizer_L2=weight_regularizer_L2
        self.bias_regularizer_L1=bias_regularizer_L1
        self.bias_regularizer_L2=bias_regularizer_L2
    
    #@ Forward Pass:
    def forward(self, inputs):
        #Remember the input value:
        self.inputs=inputs
        #@ Calculate output values from inputs, weights and biases:
        self.output=np.dot(inputs, self.weights)+self.biases

    #@ Backward Pass:
    def backward(self, dvalues):
        #Gradients on parameters:
        self.dweights=np.dot(self.inputs.T, dvalues)
        self.dbiases=np.sum(dvalues, axis=0, keepdims=True)

        #@ Gradients on regularization
        # L1 on weights
        if self.weight_regularizer_L1>0:
            dL1=np.ones_like(self.weights)
            dL1[self.weights<0]=-1
            self.dweights+= self.weight_regularizer_L1*dL1

        # L2 on weights:
        if self.weight_regularizer_L2>0:
            self.dweights+= 2*self.weight_regularizer_L2 * self.weights
        
        # L1 on bias:
        if self.bias_regularizer_L1>0:
            dL1=np.ones_like(self.biases)
            dL1[self.biases<0]=-1
            self.dbiases+= self.bias_regularizer_L1*dL1

        # L2 on bias:
        if self.bias_regularizer_L2>0:
            self.dbiases+=2*self.bias_regularizer_L2 * self.biases
        
        # Gradient on values:
        self.dinputs=np.dot(dvalues, self.weights.T)


#@ Creating Dataset:

X, y=spiral_data(samples=100, classes=3)

#@ Creating Dense Layer with 2 input features and 3 o/p values
dense1=Dense_layer(2, 3)

#@ Performing a forward pass of our training data through this layer:
dense1.forward(X)

#Printing output of few samples:
print(dense1.output[:5])