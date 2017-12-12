import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
from IPython.display import display
from timeit import default_timer as timer
from sklearn.metrics import confusion_matrix, classification_report
from h2_util import load_train_data, load_test_data, split_data
import nn_model as nn
import cnn_model as cnn
import logistic_model as logr
from model import Classifier

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.figure()
    ax = fig.add_axes([0.15, 0.1, 0.8, 0.8])
    ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title)
    ax.colorbar()
    ax.tight_layout()
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    plt.show()

    
def export_dataframe(name, df):
    """ Trivial helper function for exporting pandas data frames to the right folder """
    result_path = 'results'
    my_path = os.path.join(result_path, name)
    df.to_csv(my_path, index=False)

def export_fig(name, fig):
    """ Trivial helper function for exporting figures to the right folder """
    result_path = 'results'
    my_path = os.path.join(result_path, name)
    fig.savefig(my_path)
    
def get_digit_pair_data(X, y, i, j):
    """ Extract classes i, j from data and return a new training set with only these and labels 0, 1 for class i and j respectively """
    filt = (y == i) | (y == j)    
    X_fil = X[filt, :]
    y_fil = y[filt]
    y_fil = (y_fil == i).astype('int64')
    return X_fil, y_fil

def model_accuracy(model, X, y):
    pred = model.predict(X)
    acc = np.mean(pred == y)
    return acc

def logr_stats(model):
    print('Logistic Regression stats on 2 vs 7 - does not save anything')
    X_train, y_train =  load_train_data()
    X_test, y_test =  load_test_data()
    print('Data Loaded')
    Xbin_train, ybin_train = get_digit_pair_data(X_train, y_train, 2, 7)
    X_train, y_train, X_val, y_val = split_data(Xbin_train, ybin_train)
    print('Train Model')
    acc_train, acc_val = model.train(X_train, y_train, X_val, y_val)
    print('Model Trained')
    Xbin_test, ybin_test = get_digit_pair_data(X_test, y_test, 2, 7)
    acc_train = model_accuracy(model, Xbin_train, ybin_train)
    acc_test = model_accuracy(model, Xbin_test, ybin_test)
    print('acc train: {0}\nacc test: {1}'.format(acc_train, acc_test))

def nn_stats(model, name):
    print('Loading Data')
    X, y =  load_train_data()
    X_train, y_train, X_val, y_val = split_data(X, y)
    X_test, y_test =  load_test_data()
    print('Data Loaded')
    print('Train Model')
    acc_train, acc_val = model.train(X_train, y_train, X_val, y_val)
    print('Model Trained')
    df = pd.DataFrame(np.c_[acc_train, acc_val], columns = ['in_sample_acc', 'validation_acc'])
    export_dataframe('{0}_early_stopping.csv'.format(name), df)
    fig = plt.figure()
    ax = fig.add_axes([0.15, 0.1, 0.8, 0.8])
    n = len(acc_train)
    ax.plot(np.arange(1,n+1), 1.0 - np.array(acc_train), 'g-o', label='training error')
    ax.plot(np.arange(1,n+1), 1.0 - np.array(acc_val), 'b-o', label='validation error')
    ax.legend()
    ax.set_title('Minimize Error Progress')
    ax.set_ylabel('Error')
    ax.set_xlabel('Epoch')
    ax.set_xticks(np.arange(1,n+1))
    export_fig('{0}_early_stopping'.format(name), fig)
    model_stats(model, name)
    
def model_stats(model, name):
    """ Train a model and make classification report and confusion matrix and save them """
    X_train, y_train = load_train_data()
    X_test, y_test = load_test_data()
    acc_train = model_accuracy(model, X_train, y_train)
    acc_test = model_accuracy(model, X_test, y_test)
    df_acc = pd.DataFrame(np.c_[acc_train, acc_test], columns=['train_accuracy', 'test_accuracy'])
    print('df_acc', df_acc)
    export_dataframe('{0}_stats_accuracy.csv'.format(name.lower()), df_acc)
    print('Train Accuracy: {0}, Test Accuracy: {1}'.format(acc_train, acc_test))
    pred_test = model.predict(X_test)
    confusion = confusion_matrix(y_test, pred_test)
    cr = classification_report(y_test, pred_test)
    print('Full Model Stats')
    print('Classification Report - Test Data')
    print(cr)
    print('Confusion Matrix - Test Data')
    df_confusion = pd.DataFrame(confusion)
    display(df_confusion)
    export_dataframe('{0}_confusion_matrix.csv'.format(name.lower()), df_confusion)
    return model

    
def make_log_statistics():
    print('Make NN Statistics...')
    config = {'n_epochs': 10}
    model = Classifier(logr.LogisticRegressionModel, logr.Config(**config))
    model = logr_stats(model)
    return model

def make_nn_statistics():
    print('Make NN Statistics...')
    config = {'n_epochs': 30, 'hidden_size': 256}
    model = Classifier(nn.FeedForwardModel, nn.Config(**config))
    model = nn_stats(model, 'nn_{0}'.format(config['hidden_size']))
    return model

def make_cnn_statistics():
    print('Make CNN Statistics...')
    config = {'n_epochs': 30, 'hidden_size': 1024}
    model = Classifier(cnn.ConvolutionalModel, cnn.Config(**config))
    model = nn_stats(model, 'cnn_{0}'.format(config['hidden_size']))
    return model

def test():    
    config = {'hidden_size': 256}
    model = Classifier(nn.FeedForwardModel, nn.Config(name_suffix='THE_BEAST', **config))
    model_stats(model, 'BEAST_MODE')
    
if __name__=='__main__':
    """ Add some options with argparser here """
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    parser = argparse.ArgumentParser()
    parser.add_argument('-nn', default=False, help='Make Feed Forward Neural Net Stats', dest='nn', action='store_true')
    parser.add_argument('-cnn', default=False, help='Make Convolutional Neural Net Stats', dest='cnn', action='store_true')
    parser.add_argument('-logr', default=False, help='Make Logistic Regression Stats', dest='logr', action='store_true')
    parser.add_argument('-test', default=False, help='Run test', dest='test', action='store_true')

    if not os.path.exists('results'):
        print('create results folder')
        os.mkdir('results')
    
    args = parser.parse_args()
    if args.nn:
        make_nn_statistics()
    if args.cnn:
        model = make_cnn_statistics()
    if args.logr:
        make_log_statistics()
    if args.test:
        test()
    
