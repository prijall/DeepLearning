import  numpy as np
from Common_loss import Loss

  
class Loss_BinaryCrossentropy(Loss):
   
   #forward pass:
   def forward(self, y_pred, y_true):
      # clipping data to prevent 0
      # CLipping  both sides to not drag mean towards any value
      y_pred_clipped=np.clip(y_pred, 1e-7, 1-1e-7)
       
      # calculate sample-wise loss
      sample_losses=-(y_true * np.log(y_pred_clipped) + 
                      (1- y_true)* np.log(1-y_pred_clipped))
      sample_losses=np.mean(sample_losses, axis=-1)

      return sample_losses
   
   # Backward pass:
   def backppass(self, dvalues, y_true):

      samples=len(dvalues)
      outputs=len(dvalues[0])
      clipped_dvalues=np.clip(dvalues, 1e-7, 1-1e-7) 
      self.dinputs=-(y_true/clipped_dvalues - (1-y_true)/ (1-clipped_dvalues))/outputs
      self.dinputs=self.dinputs/samples