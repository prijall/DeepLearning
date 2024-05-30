import numpy as np
import nnfs
from nnfs.datasets import vertical_data
from SoftMax_function import Softmax_Activation
from ReLu_condition_Part2 import ReLu_Activation
from LayersofNeurons_part2 import Dense_layer
from Common_loss import Loss_CategoricalCrossentropy

#@ Creating dataset:

X, y=vertical_data(samples=100, classes=3)

#@ Creating model:
dense1=Dense_layer(2, 3)  #@ First Dense Layer having 2 inputs
activation1=ReLu_Activation()

dense2=Dense_layer(3,3) #@second Dense layer having 3 i/p and 3 o/p
activation2=Softmax_Activation()

#@ Creating loss function:
loss_function=Loss_CategoricalCrossentropy()


#@Helper Variables:
lowest_loss=99999 #@ initial value
best_dense1_weights=dense1.weights.copy()
best_dense1_biases=dense1.biases.copy()
best_dense2_weights=dense1.weights.copy()
best_dense2_biases=dense1.biases.copy()

for iteration in range(10000):

    #@ Updating the weights with small random values:
    dense1.weights+=0.05*np.random.randn(2, 3)
    dense1.biases+=0.05*np.random.randn(1, 3)
    dense2.weights+=0.05*np.random.randn(3, 3)
    dense2.biases+=0.05*np.random.randn(1, 3)

    #@ Performing a forward pass of our training data through this layer:
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    # Perform a forward pass through activation function
    # it takes the output of second dense layer here and returns loss
    loss = loss_function.calculate(activation2.output, y)

    #@ Calculating accuracy from output of activation2 and targets
    #@ Calculating values along first axis

    predictions=np.argmax(activation2.output, axis=1)
    accuracy=np.mean(predictions==y)

    # If loss is smaller - print and save weights and biases aside
    if loss < lowest_loss:
        print('New set of weights found, iteration:', iteration,
        'loss:', loss, 'acc:', accuracy)
        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases = dense1.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.biases.copy()
        lowest_loss = loss
        # Revert weights and biases
    else:
        dense1.weights = best_dense1_weights.copy()
        dense1.biases = best_dense1_biases.copy()
        dense2.weights = best_dense2_weights.copy()
        dense2.biases = best_dense2_biases.copy()