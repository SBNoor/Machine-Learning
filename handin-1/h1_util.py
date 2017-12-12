import os
import numpy as np

class MlModel():
    """ Simple Machine Learning Model Class """
    
    def __init__(self):
        pass

    def train(self, X, y):
        raise NotImplementedError

    def visualize_model(self):
        raise NotImplementedError

    def predict(self, X):
        """ Compute the predictions on data X
        
        Args:
        X: np.array shape (n,d)
        Returns:
        pred: np.array  shape (n,) dtype = int64
        """
        raise NotImplementedError

    def probability(self, X):
        """ Compute the proability of the classes for each input point
        
        """
        raise NotImplementedError
        
    
def load_train_data():
    """ Load and return the training data """
    if not os.path.exists('auTrain.npz'):
        os.system('wget https://users-cs.au.dk/jallan/ml/data/auTrain.npz')
    tmp = np.load('auTrain.npz')
    au_digits = tmp['digits']
    print('shape of input data', au_digits.shape)
    au_labels = np.squeeze(tmp['labels'])
    print('labels shape and type', au_labels.shape, au_labels.dtype)
    return au_digits, au_labels

def load_test_data():
    """ Load and return the test data """
    filename = 'auTest.npz';
    if not os.path.exists('auTest.npz'):
        os.system('wget https://users-cs.au.dk/jallan/ml/data/auTest.npz')        
    tmp = np.load('auTest.npz')
    au_digits = tmp['digits']
    print('shape of input data', au_digits.shape)
    au_labels = np.squeeze(tmp['labels'])
    print('labels shape and type', au_labels.shape, au_labels.dtype)
    return au_digits, au_labels

def numerical_grad_check(f, x):
    """ Numerical Gradient Checker """
    eps = 1e-6
    h = 1e-4
    d = x.shape[0]
    cost, grad = f(x)
    it = np.nditer(x, flags=['multi_index'])
    while not it.finished:
        dim = it.multi_index
        tmp = x[dim]
        x[dim] = tmp + h
        cplus, _ = f(x)
        x[dim] = tmp - h 
        cminus, _ = f(x)
        x[dim] = tmp
        num_grad = (cplus-cminus)/(2*h)
        print('grad, num_grad, grad-num_grad', grad[dim], num_grad, grad[dim]-num_grad)
        assert np.abs(num_grad - grad[dim]) < eps, 'numerical gradient error index {0}, numerical gradient {1}, computed gradient {2}'.format(dim, num_grad, grad[dim])
        it.iternext()

    
            
            
    
