import numpy as np
from LayersofNeurons_part2 import Dense_layer
from ReLu_condition_Part2 import ReLu_Activation 
from  SoftMax_function import Softmax_Activation
from Common_loss import Loss, Loss_CategoricalCrossentropy
from Activation_Softmax_Loss_CategoricalCrossentropy import Activation_Softmax_Loss_CategoricalCrossentropy

import nnfs

from nnfs.datasets import spiral_data
nnfs.init()

#creating dataset:
X, y=spiral_data(samples=100, classes=3)

#@ Creating Dense Layer with 2 i/p and 3 o/p features:
dense1=Dense_layer(2, 3)

#@Creating ReLU activation
activation1=ReLu_Activation()

# Create second Dense layer with 3 input features (as we take output
# of previous layer here) and 3 output values (output values)
dense2 = Dense_layer(3, 3)

# Create Softmax classifier's combined loss and activation
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

#@ Performing forward pass of training data:
dense1.forward(X)

#@Performing forward pass through activation function takes the output of first dense layer:
activation1.forward(dense1.output)

# Perform a forward pass through second Dense layer
# takes outputs of activation function of first layer as inputs
dense2.forward(activation1.output)

# Perform a forward pass through the activation/loss function
# takes the output of second dense layer here and returns loss
loss = loss_activation.forward(dense2.output, y)

# Let's see output of the first few samples:
print(loss_activation.output[:5])

# Print loss value
print('loss:', loss)

# Calculate accuracy from output of activation2 and targets
# calculate values along first axis
predictions = np.argmax(loss_activation.output, axis=1)
if len(y.shape) == 2:
 y = np.argmax(y, axis=1)
accuracy = np.mean(predictions==y)

# Print accuracy
print('acc:', accuracy)

# Backward pass
loss_activation.backward(loss_activation.output, y)
dense2.backward(loss_activation.dinputs)
activation1.backward(dense2.dinputs)
dense1.backward(activation1.dinputs)

# Print gradients
print(dense1.dweights)
print(dense1.dbiases)
print(dense2.dweights)
print(dense2.dbiases)