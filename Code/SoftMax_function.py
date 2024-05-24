import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from LayersofNeurons_part2 import Dense_layer
from ReLu_condition_Part2 import ReLu_Activation

nnfs.init()

#@ Softmax Activation:
class Softmax_Activation:

    #Forward Pass:
     def forward(self, inputs):
        #@ Getting Unnormaalized probabilities:
        exp_values=np.exp(inputs- np.max(inputs, axis=1, keepdims=True))

        #@ Normalizing them for each sample:
        probabilities=exp_values/np.sum(exp_values, axis=1, keepdims=True)

        self.output=probabilities
    
# Creating dataset
X, y = spiral_data(samples=100, classes=3)

# Creating Dense layer with 2 input features and 3 output values
dense1 = Dense_layer(2, 3)

# Creating ReLU activation (to be used with Dense layer):
activation1 = ReLu_Activation()

# Creating second Dense layer with 3 input features (as we take output
# of previous layer here) and 3 output values (output values)
dense2 = Dense_layer(3, 3)

# Creating  Softmax activation (to be used with Dense layer):
activation2 =Softmax_Activation()

#@ Creating Softmax activation(to be used with dense layer):
dense1.forward(X)

# Making forward pass through activation function
# it takes the output of first dense layer here
activation1.forward(dense1.output)

# Making forward pass through second Dense layer
# it takes outputs of activation function of first layer as inputs
dense2.forward(activation1.output)

# Making forward pass through activation function
# it takes the output of second dense layer here
activation2.forward(dense2.output)

# Let's see output of the first few samples:
print(activation2.output[:5])


