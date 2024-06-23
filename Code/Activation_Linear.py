class Linear_Activation:

    #forward pass:
    def forward(self, inputs):
        self.inputs=inputs
        self.output=inputs

    #Backward Pass
    def backward(self, dvalues):
        #derivative is 1, 1* dvalues - dvalues - the chain rule
        self.dinputs=dvalues.copy()

    # adding prediction for output:
    def predictions(self, outputs):
          return outputs

#@Example:

# import numpy as np

# # Example inputs
# inputs = np.array([[1.0, 2.0, 3.0],
#                    [2.0, 5.0, -1.0],
#                    [-1.5, 2.7, 3.3]])

# # Instantiate the linear activation function
# activation = Linear_Activation()

# # Forward pass
# activation.forward(inputs)
# #print("Forward pass output:")
# print(activation.output)

# # Example gradient (usually coming from the next layer during backpropagation)
# dvalues = np.array([[1.0, 1.0, 1.0],
#                     [2.0, 2.0, 2.0],
#                     [3.0, 3.0, 3.0]])

# # Backward pass
# activation.backward(dvalues)
# #print("Backward pass output:")
# print(activation.dinputs)
