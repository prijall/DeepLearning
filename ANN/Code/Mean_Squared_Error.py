import numpy as np
from Common_loss import Loss

class Loss_MeanSquaredError(Loss):
     
     #forward pass:

     def forward(self, y_pred, y_true):
          
          # Calculating loss
          sample_losses=np.mean((y_true-y_pred)**2, axis=1)
          return sample_losses
     
     def backward(self, dvalues, y_true):
          
        samples=len(dvalues)
        outputs=len(dvalues[0])

        self.dinputs=-2 * (y_true-dvalues)/outputs
        self.dinputs=self.dinputs/ samples




# # Example predicted values
# y_pred = np.array([[0.1, 0.2, 0.3],
#                    [0.4, 0.5, 0.6],
#                    [0.7, 0.8, 0.9]])

# # Example true values
# y_true = np.array([[0.0, 0.0, 0.0],
#                    [1.0, 1.0, 1.0],
#                    [0.5, 0.5, 0.5]])

# # Instantiate the MSE loss
# loss = Loss_MeanSquaredError()

# # Forward pass (calculating the loss)
# sample_losses = loss.forward(y_pred, y_true)
# print("Sample losses:")
# print(sample_losses)

# # Example dvalues (usually coming from the next layer during backpropagation)
# dvalues = y_pred  # In practice, these would be gradients from the next layer

# # Backward pass (calculating the gradient of the loss with respect to predictions)
# loss.backward(dvalues, y_true)
# print("Gradients of the loss with respect to predictions:")
# print(loss.dinputs)
