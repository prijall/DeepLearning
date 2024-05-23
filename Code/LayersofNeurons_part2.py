import numpy as np
import nnfs
from nnfs.datasets import spiral_data



nnfs.init()

class Dense_layer:

    #@ Layers Initialization:
    def __init__(self, n_inputs, n_neurons):
        #@ Initializing weights and bias:
        self.weights=0.01*np.random.randn(n_inputs, n_neurons)
        self.biases=np.zeros((1, n_neurons))
    
    #@ Forward Pass:
    def forward(self, inputs):
        #@ Calculate output values from inputs, weights and biases:
        self.output=np.dot(inputs, self.weights)+self.biases


#@ Creating Dataset:

X, y=spiral_data(samples=100, classes=3)

#@ Creating Dense Layer with 2 input features and 3 o/p values
dense1=Dense_layer(2, 3)

#@ Performing a forward pass of our training data through this layer:
dense1.forward(X)

#Printing output of few samples:
print(dense1.output[:5])