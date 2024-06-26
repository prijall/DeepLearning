import numpy as np

class Dropout_Layer:

    def __init__(self, rate):
        self.rate=1-rate

    #@ Forward Pass:
    def forward(self, inputs, training):
        # Saving the input values:
        self.inputs=inputs

        # if not in training mode, return values:
        if not training:
            self.output=inputs.copy()
            return


        #Generating and save scaled mask:
        self.binary_mask=np.random.binomial(1, self.rate, size=inputs.shape)/ self.rate

        # Applying mask to o/p values:
        self.output=inputs*self.binary_mask

    #@ Backward Pass:
    def backward(self, dvalues):
        #Gradient on values:
        self.dvalues=dvalues*self.binary_mask


# # Initialize dropout layer with 20% dropout rate
# dropout = Dropout_Layer(rate=0.2)

# # Example input array
# inputs = np.array([0.27, -1.03, 0.67, 0.99, 0.05, -0.37, -2.01, 1.13, -0.07, 0.73])

# # Forward pass
# dropout.forward(inputs)
# print("Output after dropout forward pass:", dropout.output)

# # Example gradients from the next layer in backpropagation
# dvalues = np.ones_like(inputs)

# # Backward pass
# dropout.backward(dvalues)
# print("Gradients after dropout backward pass:", dropout.dvalues)
