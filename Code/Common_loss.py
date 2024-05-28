import numpy as np

class Loss:
    #@ Calculates the data and regularization losses
    def calculate(self, output, y):

        #@ Calculate sample losses:
        sample_losses=self.forward(output, y)

        #@ Calculate mean loss
        data_loss=np.mean(sample_losses)

        #@ Return loss
        return data_loss
    
#@ Cross-Entropy Loss:

class Loss_CategoricalCrossEntropy(Loss):

    #@ Forward Pass:
    def forward(self, y_pred, y_true):

        #@ Number of samples in a batch:
        samples=len(y_pred)

        #@ Clipping data to prevent division by 0, Clipping both sides to not drag mean towards any value
        y_pred_clipped=np.clip(y_pred, 1e-7, 1-1e-7)
         
        #@ Probabilities for target values only if there is categorical labels:
        if len(y_pred.shape)==1:
            correct_confidences=y_pred_clipped[range(samples), y_true]
        
        #@ Mask values- only for one hot encoded labels:
        elif len(y_true.shape)==2:
            correct_confidences=np.sum(y_pred_clipped*y_true, axis=1)

        #@ Losses:
        negative_log_likelihoods=-np.log(correct_confidences)
        return negative_log_likelihoods