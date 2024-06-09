import numpy as np
from LayersofNeurons_part2 import Dense_layer
from ReLu_condition_Part2 import ReLu_Activation 
from  SoftMax_function import Softmax_Activation
from Common_loss import Loss, Loss_CategoricalCrossentropy
from Activation_Softmax_Loss_CategoricalCrossentropy import Activation_Softmax_Loss_CategoricalCrossentropy

from Activation_Softmax_Loss_CategoricalCrossentropy import Activation_Softmax_Loss_CategoricalCrossentropy

import nnfs

from nnfs.datasets import spiral_data
nnfs.init()

#@ Creating SDG Optimizer:
class SGD_Optimizer:
    def __init__(self, learning_rate=0.85):
        self.learning_rate=learning_rate

    #@ Updating Parameters:
    def update_params(self, layer):
        layer.weights += -self.learning_rate*layer.dweights
        layer.biases+= -self.learning_rate*layer.dbiases


# Create dataset
X, y = spiral_data(samples=100, classes=3)
# Create Dense layer with 2 input features and 64 output values
dense1 = Dense_layer(2, 64)
# Create ReLU activation (to be used with Dense layer):
activation1 = ReLu_Activation()
# Create second Dense layer with 64 input features (as we take output
# of previous layer here) and 3 output values (output values)
dense2 = Dense_layer(64, 3)
# Create Softmax classifier's combined loss and activation
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
# Create optimizer
optimizer = SGD_Optimizer()
# Train in loop
for epoch in range(10001):
# Perform a forward pass of our training data through this layer
     dense1.forward(X)
     activation1.forward(dense1.output)
     dense2.forward(activation1.output)
     loss = loss_activation.forward(dense2.output, y)
  
     predictions = np.argmax(loss_activation.output, axis=1)
     if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
     accuracy = np.mean(predictions==y)
     if not epoch % 100:
           print(f'epoch: {epoch}, ' +
             f'acc: {accuracy:.3f}, ' +
             f'loss: {loss:.3f}')
       
# Backward pass
loss_activation.backward(loss_activation.output, y)
dense2.backward(loss_activation.dinputs)
activation1.backward(dense2.dinputs)
dense1.backward(activation1.dinputs)
# Update weights and biases
optimizer.update_params(dense1)
optimizer.update_params(dense2)