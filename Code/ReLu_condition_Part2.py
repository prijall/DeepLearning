#@ Importing dependencies:
import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from LayersofNeurons_part2 import Dense_layer

#@ RELU Activation:
class ReLu_Activation:
    # forward pass
    def forward(self, inputs):
        #Remember the input value:
        self.inputs=inputs
        #calculate output values from input:
        self.output=np.maximum(0, inputs)

    # Backward pass:
    def backward(self, dvalues):
        #before modifying original variable, let's make a copy
        self.dinputs=dvalues.copy()

        #Zero gradient where input values were negative:
        self.dinputs[self.inputs<=0]=0

#@ Creating a dataset:
X, y=spiral_data(samples=100, classes=3)

#@ Creating a dense layer with 2 input features and 3 output value
dense1=Dense_layer(2, 3)

#@ Creating RELU Activation:
activation1=ReLu_Activation()
dense1.forward(X)
activation1.forward(dense1.output)

print(activation1.output[:5])