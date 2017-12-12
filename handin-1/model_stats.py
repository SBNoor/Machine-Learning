import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
from IPython.display import display
from timeit import default_timer as timer
from sklearn.metrics import confusion_matrix, classification_report
from h1_util import load_train_data, load_test_data
import logistic_regression as log_reg
from h1_classifiers import LogisticClassifierTwoClass, LogisticClassifier, SoftmaxClassifier, model_accuracy, run_validation

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def get_digit_pair_data(X, y, i, j):
    """ Extract classes i, j from data and return a new training set with only these and labels 0, 1 for class i and j respectively """
    filt = (y == i) | (y == j)    
    X_fil = X[filt, :]
    y_fil = y[filt]
    y_fil = (y_fil == i).astype('int64')
    return X_fil, y_fil

def export_dataframe(name, df):
    result_path = 'results'
    my_path = os.path.join(result_path, name)
    df.to_csv(my_path, index=False)

def export_fig(name, fig):
    result_path = 'results'
    my_path = os.path.join(result_path, name)
    fig.savefig(my_path)
    

def model_stats(model, **kwargs):
    """ Train a model and make classification report and confusion matrix and save them """
    X_train, y_train =  load_train_data()
    X_test, y_test =  load_test_data()
    model.train(X_train, y_train, **kwargs) 
    acc_train = model_accuracy(model, X_train, y_train)
    acc_test = model_accuracy(model, X_test, y_test)
    df_acc = pd.DataFrame(np.c_[acc_train, acc_test], columns=['train_accuracy', 'test_accuracy'])
    export_dataframe('{0}_stats_accuracy.csv'.format(model.name.lower()), df_acc)
    print('Train Accuracy: {0}, Test Accuracy: {1}'.format(acc_train, acc_test))
    pred_test = model.predict(X_test)
    confusion = confusion_matrix(y_test, pred_test)
    cr = classification_report(y_test, pred_test)
    print('Full Model Stats')
    print('Classification Report')
    print(cr)
    print('Confusion Matrix')
    df_confusion = pd.DataFrame(confusion)
    print(confusion)
    export_dataframe('{0}_confusion_matrix.csv'.format(model.name.lower()), df_confusion)
    # plot_confusion_matrix(confusion)
    return model

def make_logreg_statistics():
    """ Generate simple statistics about the logistic regression implementation for the hand in report """

    pd.set_option('display.height', 1000)
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    model = LogisticClassifierTwoClass()
    rounds = 50
    print('Loading Data')
    X_train, y_train =  load_train_data()
    X_test, y_test =  load_test_data()
    
    names = ['Batch Gradient Descent', 'Mini Batch Gradient Descent', 'Scipy Optimize']
    algs = [log_reg.batch_grad_descent, log_reg.mini_batch_grad_descent, log_reg.fast_descent]
    params = [{'reg': 1e-4, 'rounds':rounds, 'lr':1.0},
              {'reg': 1e-4, 'epochs':rounds, 'lr':0.1, 'batch_size':16},
              {'reg': 1e-4, 'rounds':rounds}]
    times = np.zeros(3)
    acc_train = np.zeros(3)
    acc_test = np.zeros(3)

    print('Computing 2 vs 7 Statistics on fixed paramters')
    Xbin_train, ybin_train = get_digit_pair_data(X_train, y_train, 2, 7)
    Xbin_test, ybin_test = get_digit_pair_data(X_test, y_test, 2, 7)
    Xbin_train_padded = np.c_[np.zeros(Xbin_train.shape[0]), Xbin_train]
    for i, (alg, param, name) in enumerate(zip(algs, params, names)):
        print('*** Running {0} ***'.format(name))
        start = timer()
        w = alg(Xbin_train_padded, ybin_train, **param)
        end = timer()
        time = end - start
        model.w = w
        train_acc = model_accuracy(model, Xbin_train, ybin_train)
        test_acc = model_accuracy(model, Xbin_test, ybin_test)
        times[i] = time
        acc_train[i] =  train_acc
        acc_test[i] =  test_acc
        print('Alg: {0}, Rounds/Epochs: {1}, Time: {2}, Ein: {3}, Etest: {4}'.format(name, rounds, time, train_acc, test_acc))
    df_2vs7 = pd.DataFrame(np.c_[names, times, acc_train,acc_test], columns=['Alg.', 'Time', 'Ein', 'Etest'])
    filename_2vs7 = 'results_2_vs_7.csv'
    print('Result of 2 vs 7 Classifier with fixed parameters stored in {0}'.format(filename_2vs7))
    display(df_2vs7)
    export_dataframe(filename_2vs7, df_2vs7)        
    visualize_model(LogisticClassifierTwoClass(), Xbin_train, ybin_train)
            
    print('Compute Pairwise scores using mini_batch gradient descent')
    pairwise_scores_train = np.zeros((10,10))
    pairwise_scores_test = np.zeros((10,10))
    pairs = [(i,j) for i in range(10) for j in range(10)]
    model = LogisticClassifierTwoClass()
    for i, j in pairs:
        if j >= i: continue
        Xbin_train, ybin_train = get_digit_pair_data(X_train, y_train, i, j)
        Xbin_test, ybin_test = get_digit_pair_data(X_test, y_test, i, j)        
        # model.w = log_reg.mini_batch_grad_descent(Xbin_train, ybin_train, reg=1e-4, epochs=30, lr=0.1, batch_size=16)
        model.train(Xbin_train, ybin_train, reg=1e-4, epochs=30, lr=0.1, batch_size=16)
        acc_train = model_accuracy(model, Xbin_train, ybin_train)
        acc_test = model_accuracy(model, Xbin_test, ybin_test)
        pairwise_scores_train[i][j] = acc_train
        pairwise_scores_test[i][j] = acc_test
    
    df_pairwise_train = pd.DataFrame(pairwise_scores_train)
    df_pairwise_test = pd.DataFrame(pairwise_scores_test)
    print('Pairwise Training Accuracy')
    display(df_pairwise_train)
    print('Pairwise Test Accuract')
    display(df_pairwise_test)
    export_dataframe('pairwise_train_accuracy.csv', df_pairwise_train)
    export_dataframe('pairwise_test_accuracy.csv', df_pairwise_test)
    
    print('Training full model with all vs one')
    full_model = LogisticClassifier()
    model_stats(full_model, **{'reg': 1e-4, 'lr': 0.1, 'epochs': 30, 'batch_size': 16})
    print('Result of 2 vs 7 Classifier with fixed parameters stored in {0}'.format(filename_2vs7))
    visualize_model(LogisticClassifier())


def make_softmax_statistics():
    model = SoftmaxClassifier()
    rounds = 50
    print('Loading Data')
    X_train, y_train = load_train_data()
    X_test, y_test = load_test_data()
    model = model_stats(model, **{'reg': 1e-4, 'epochs': 1, 'batch_size': 64})
    visualize_model(SoftmaxClassifier())
    return model
                        

def visualize_model(model, X_train=None, y_train=None):
    """ Visualize a model after training for at short time 

    Args:
     model: MlModel 
     X_train: numpy array, training data
     y_train: numpy array, training labels
    """
    p1 = {'reg': 1e-4, 'epochs': 1, 'batch_size': 128}
    # p2 = {'reg': 1e-4, 'epochs': 1, 'batch_size': 16}    
    # p3 = {'reg': 1e-4, 'epochs': 10, 'batch_size': 16}
    if X_train is None:
        X_train, y_train = load_train_data()
    model.train(X_train, y_train, **p1)
    fig = plt.figure()
    # ax = fig.add_axes([0.15, 0.1, 0.7, 0.3])
    ax = fig.add_subplot(1,1,1)
    model.visualize_model(ax)
    export_fig('{0}_parameter_plot_{1}_{2}'.format(model.name.lower(), p1['epochs'], p1['batch_size']), fig)
    # plt.show()


def best_model(model):
    """ Find the best regularization of the model
    All you need to do is put some regularization values in reg and
    choose the params you want for mini_batch_gradient descent in params.
    """

    X_train, y_train =  load_train_data()
    X_test, y_test =  load_test_data()
    params = {}
    reg = []
    ### YOUR CODE HERE
    reg = [0.2**i for i in range(10)]
    ### END CODE
    reg = sorted(reg)
    model, acc = run_validation(model, X_train, y_train, reg, **params)
    df_reg = pd.DataFrame(np.c_[reg, acc], columns=['reg. params', 'val. score'])
    export_dataframe('{0}_validation_scores.csv'.format(model.name.lower()), df_reg)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)    
    ax.plot(reg, acc, 'b-*', linewidth=2, markersize=6, label='Validation Accuracy')
    print('plot of: ', np.c_[reg, acc])
    name = '{0}_validation_accuracy'.format(model.name.lower())
    ax.set_title(name)
    ax.set_xlabel('reg. params')
    ax.set_ylabel('val. accuracy')
    ax.legend()    
    export_fig('{0}.png'.format(name), fig)
    test_acc = model_accuracy(model, X_test, y_test)
    idx = np.argmax(acc)    
    print('Best model found: reg {0}\nValidation accuracy: {1}\nTest Accuracy: {2}'.format(reg[idx], acc[idx], test_acc))
    best_result = pd.DataFrame(np.c_[reg[idx], acc[idx], test_acc], columns=['reg. value', 'validation accuracy', 'test accuracy'])
    export_dataframe('{0}_best_result.csv'.format(model.name.lower()), best_result)

if __name__=='__main__':
    """ Add some options with argparser here """
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    parser = argparse.ArgumentParser()
    parser.add_argument('-log', default=False, help='Make Logistic Regression Stats', dest='log', action='store_true')
    parser.add_argument('-soft', default=False, help='Make Softmax Stats', dest='soft', action='store_true')
    parser.add_argument('-log_val', default=False, help='Make Best Logistic Regression', dest='log_val', action='store_true')
    parser.add_argument('-soft_val', default=False, help='Make Best Softmax ', dest='soft_val', action='store_true')

    # visualize_model(LogisticClassifier())
    if not os.path.exists('results'):
        print('create results folder')
        os.mkdir('results')
    
    
    vals = vars(parser.parse_args())
    if vals['log']:        
        make_logreg_statistics()
    if vals['soft']:
        model = make_softmax_statistics()
    if vals['log_val']:
        best_model(LogisticClassifier())
    if vals['soft_val']:
        best_model(SoftmaxClassifier())
 
