import numpy as np

def stable_softmax(x)->float:
    if x.ndim==1:
        shifted_x=x-np.max(x)
        exp_val=np.exp(shifted_x)
        return exp_val/np.sum(exp_val)
    else:
        shifted_x=x-np.max(x, axis=1, keepdims=True)
        exp_val=np.exp(shifted_x)
        exp_sum=np.sum(exp_val, axis=1, keepdims=True)
        return exp_val / exp_sum
    

def softmax_cross_entropy_loss(logits, labels):
    calculate_probabilities=stable_softmax(logits)
    print('Calculate Probabilities:', calculate_probabilities)
    print('\n')

    batch_indices=np.arange(logits.shape[0])
    corrected_log_probabilities=-np.log(calculate_probabilities[batch_indices, labels])
    print('Corrected probabilities', corrected_log_probabilities)
    print('\n')

    return np.mean(corrected_log_probabilities)



logits = np.array([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]])
labels = np.array([0, 1])

loss=softmax_cross_entropy_loss(logits, labels)
print('cross entropy loss', f'{loss:.2f}')