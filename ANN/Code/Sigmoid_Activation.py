import numpy as np

class Activation_Sigmoid:

     # Forward Pass:
     def forward(self, inputs):
          self.inputs=inputs
          self.output= 1/ (1 +  np.exp(-inputs))

     # Backward Pass:
     def backward(self, dvalues):
          
          self.dinputs=dvalues * (1- self.output) * self.output
     
     # adding prediction for output:
     def predictions(self, outputs):
          return (outputs>0.5)*1