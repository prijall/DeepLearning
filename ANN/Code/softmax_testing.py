import numpy as np

#@Probabilities of 3 samples:
softmax_output= np.array([[0.7, 0.2, 0.1],
                          [0.5, 0.1, 0.4],
                          [0.02, 0.9, 0.08]])

#@ Target labels for 3 samples:
class_targets=np.array([0, 1, 1])

#@ Calculating values along second axis(axis of index 1)
predictions=np.argmax(softmax_output, axis=1)

#@ If targets are one-hot encoded - Convert them:
if len(class_targets.shape)==2:
    class_targets=np.argmax(class_targets, axis=1)

#@ True evaluates to 1; False to 0
accuracy = np.mean(predictions==class_targets)

print("Accuracy:", accuracy)


