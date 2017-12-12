import os
import numpy as np

def load_train_data():
    if not os.path.exists('auTrain.npz'):
        os.system('wget https://users-cs.au.dk/jallan/ml/data/auTrain.npz')
    tmp = np.load('auTrain.npz')
    return tmp['digits'], tmp['labels']

def load_test_data():
    if not os.path.exists('auTest.npz'):
        os.system('wget https://users-cs.au.dk/jallan/ml/data/auTest.npz')
    tmp = np.load('auTest.npz')
    return tmp['digits'], tmp['labels']

def split_data(X_, y_, percentage=0.2):
    """ Splits the data into a training and a validation set """
    n = y_.shape[0]
    val_size = int(n * percentage)
    rp = np.random.permutation(y_.shape[0])
    X = X_[rp]
    y = y_[rp]       
    val_train = X[0:val_size,:]
    val_target = y[0:val_size]
    train = X[val_size+1:,:]
    target = y[val_size+1:]
    return train, target, val_train, val_target

