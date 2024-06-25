#@ Softmax Classifier - combined Softmax activation 
# and cross- entropy loss for faster backward step:
import numpy as np
from SoftMax_function import Softmax_Activation
from Common_loss import Loss_CategoricalCrossentropy

class Activation_Softmax_Loss_CategoricalCrossentropy():
    #@ Creating activation and loss function objects:
    def __init__(self):
        self.activation=Softmax_Activation()
        self.loss=Loss_CategoricalCrossentropy()

    #@ Forward Pass:
    def forward(self, inputs, y_true):
        # Output layer's activation function:
        self.activation.forward(inputs)

        #set the output:
        self.output=self.activation.output

        # Calcuate and return loss:
        return self.loss.calculate(self.output, y_true)
    
    #@ Backward Pass:
    def backward(self, dvalues, y_true):

        #Number of samples:
        samples=len(dvalues)

        #If labels are one-hot encoded,
        #turn them into discrete values:

        if len(y_true.shape)==2:
            y_true=np.argmax(y_true, axis=1)

        #copy for safe modification:
        self.dinputs=dvalues.copy()

        #calculate gradient
        self.dinputs[range(samples), y_true]-=1
        
        #Normalize gradient:
        self.dinputs= self.dinputs/samples
