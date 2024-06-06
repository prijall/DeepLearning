import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from SoftMax_function import Softmax_Activation
from ReLu_condition_Part2 import ReLu_Activation
from LayersofNeurons_part2 import Dense_layer

nnfs.init()


# Common loss class
class Loss:
# Calculates the data and regularization losses
# given model output and ground truth values
 def calculate(self, output, y):

  sample_losses = self.forward(output, y)
  data_loss = np.mean(sample_losses)
  return data_loss
 
# Cross-entropy loss
class Loss_CategoricalCrossentropy(Loss):
# Forward pass
   def forward(self, y_pred, y_true):
     samples = len(y_pred)
     y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

     if len(y_true.shape) == 1:
        correct_confidences = y_pred_clipped[
        range(samples),
        y_true
          ]
# Mask values - only for one-hot encoded labels
     elif len(y_true.shape) == 2:
       correct_confidences = np.sum(
       y_pred_clipped * y_true,
       axis=1
          )

     negative_log_likelihoods = -np.log(correct_confidences)
     return negative_log_likelihoods
   

   def backward(self, dvalues, y_true):
     #number of samples
     samples=len(dvalues)

     #number of labels in every sample
     # we'll use the first sample to count them
     labels=len(dvalues[0])

     # If labels are sparse, turn them into one-hot vector:
     if len(y_true.shape)==1:
       y_true=np.eye(labels)[y_true]

    # Calculate gradient:
     self.dinputs=-y_true / dvalues

     # Normalize gradient:
     self.dinputs= self.dinputs / samples



     

# Create dataset
X, y = spiral_data(samples=100, classes=3)
# Create Dense layer with 2 input features and 3 output values
dense1 = Dense_layer(2, 3)
# Create ReLU activation (to be used with Dense layer):
activation1 = ReLu_Activation()
# Create second Dense layer with 3 input features (as we take output
# of previous layer here) and 3 output values
dense2 = Dense_layer(3, 3)
# Create Softmax activation (to be used with Dense layer):
activation2 = Softmax_Activation()
# Create loss function
loss_function = Loss_CategoricalCrossentropy()
# Perform a forward pass of our training data through this layer
dense1.forward(X)
# Perform a forward pass through activation function
# it takes the output of first dense layer here
activation1.forward(dense1.output)

# Perform a forward pass through second Dense layer
# it takes outputs of activation function of first layer as inputs
dense2.forward(activation1.output)
# Perform a forward pass through activation function
# it takes the output of second dense layer here
activation2.forward(dense2.output)
# Let's see output of the first few samples:
print(activation2.output[:5])
# Perform a forward pass through loss function
# it takes the output of second dense layer here and returns loss
loss = loss_function.calculate(activation2.output, y)
# Print loss value
print('loss:', loss)