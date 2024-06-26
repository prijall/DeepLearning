import numpy as np

class Accuracy:
    #calculates an accuracy given predictions and ground truth values:
    def calculate(self, predictions, y):
        #getting comparison results:
        comparisons=self.compare(predictions, y)

        #calculate accuracy:
        accuracy=np.mean(comparisons)
        return accuracy
    

#@ Accuracy calcualtion for regresison model:

class Accuracy_Regression(Accuracy):

    def __init__(self):
        #Creating precision Property:
        self.precision=None
        
    #calcualting precision value based on passed in ground truth:
    def init(self, y, reinit=False):
        if self.precision is None or reinit:
            self.precision =np.std(y)/250

    # Compare predictions to ground truth value:
    def comapre(self, predictions, y):
        return np,abs(predictions-y)<self.precision 
    

#@ Accuracy calculation for classification model:

class Accuracy_Categorical(Accuracy):
    # No initialization is needed:
    def init(self, y):
        pass

    #comparing predictions to the ground truth:
    def compare(self, predictions, y):
        if len(y.shape)==2:
            y=np.argmax(y, axis=1)
        return predictions==y