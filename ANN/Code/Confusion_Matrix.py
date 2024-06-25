#@ Importing all libraries and dependencies:
import numpy as np
from sklearn.metrics import confusion_matrix,classification_report
import seaborn as sns
import matplotlib.pyplot as plt 

#@ Creating dataset for Predicted and Actual values:
actual    = np.array(
  ['Dog','Dog','Dog','Not Dog','Dog','Not Dog','Dog','Dog','Not Dog','Not Dog'])
predicted = np.array(
  ['Dog','Not Dog','Dog','Not Dog','Dog','Dog','Dog','Dog','Not Dog','Not Dog'])


#@ Creating Object for Confusing matrix and pass both predicted and true values
matix=confusion_matrix(actual, predicted)

#@ Visualizing:

sns.heatmap(matix, annot=True, fmt='g',
            xticklabels=['Dog', 'Not Dog'], 
            yticklabels=['Dog', 'Not Dog'],)
plt.xlabel('Prediction', fontsize=13)
plt.ylabel('Actual', fontsize=13)
plt.title('Confusion Matrix',fontsize=17)
plt.show()

print(classification_report(actual, predicted))