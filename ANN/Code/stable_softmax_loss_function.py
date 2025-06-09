import numpy as np

def stable_softmax(x):
    if x.ndim==1:
        x_shifted=x-np.max(x)
        exp_x=np.exp(x_shifted)
        sum_x=np.sum(exp_x)
        return exp_x/sum_x
    
    else:
        x_shifted=x-np.max(x, axis=1, keepdims=True)
        exp_x=np.exp(x_shifted)
        sum_x=np.sum(exp_x)
        return exp_x/sum_x
    

def softmax_cross_entropy_loss(logits, labels):
    compute_probab=stable_softmax(logits)
    print('Compute Probability:\n', compute_probab)

    batch_indices=np.arange(logits.shape[0])
    corrected_logprobaabilities=-np.log(compute_probab[batch_indices, labels])

    print('correct Log Probabilites:\n', corrected_logprobaabilities)

    return np.mean(corrected_logprobaabilities)
 

logits1=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
labels1=np.array([0,1])

loss=softmax_cross_entropy_loss(logits1, labels1)
print(f'softmax cross entropy loss:{loss:.2f}\n')