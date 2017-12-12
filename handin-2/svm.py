from sklearn import svm
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
import os
import argparse
from IPython.display import display
from h2_util import load_train_data, load_test_data
from model_stats import export_dataframe, model_stats


def save_model(cv, name):
    """ Simple save the SVC object learned with cross validation, ignore if you do not need it 
    
    Params:
      cv: GridSearchCV fitted on data
      name: str - name to give model, i.e. rbf_kernel or something like that
    """
    print('saving', name)
    np.savez_compressed(os.path.join('model_weights','{0}_best_model.npz'.format(name)), res=cv.best_estimator_)

def load_model(name):
    """ Simple function to load an SVM  model, ignore if you do not need it         

    Args:
      name: str - name given to model when saved
    """
    tmp = np.load(os.path.join('model_weights','{0}_best_model.npz'.format(name)))
    return tmp['res'].item()


def train(kern, X, Y):
    clf = svm.SVC(kernel=kern,C=1, decision_function_shape='over')
    print("Training...")
    clf.fit(X,Y)
    print("DONE!")
    return clf

def predict(model, X):
    return model.predict(X)

def svc_cv_lin(X, Y):
    clf = svm.SVC(kernel='linear')
    parameters = {
        'C': [0.001, 0.01, 0.1, 1, 5, 10, 50, 100],
    }
    gs_clf = GridSearchCV(clf, parameters, n_jobs=-1) 
    gs_clf = gs_clf.fit(X, Y)   
    return gs_clf.cv_results_

def svc_cv_poly(X, Y, d):
    clf = svm.SVC(kernel='poly', degree=d)
    parameters = {
        'C': [1, 5, 10, 50, 100], 'coef0': [1,2,3,4,5]
    }
    gs_clf = GridSearchCV(clf, parameters, n_jobs=-1) 
    gs_clf = gs_clf.fit(X, Y)   
    return gs_clf.cv_results_

def svc_cv(X, Y):
    clf = svm.SVC(kernel='rbf')
    parameters = {
        'C': [1, 10, 100],
        'gamma': [0.0001, 0.001, 0.01, 0.1, 1]
    }
    gs_clf = GridSearchCV(clf, parameters, n_jobs=-1) 
    gs_clf = gs_clf.fit(X, Y)   
    return gs_clf.cv_results_
### END CODE

if __name__=="__main__":
    
    if not os.path.exists('results'):
        print('create results folder')
        os.mkdir('results')
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    [au_train_images,au_train_labels] = load_train_data()
    [au_test_images,au_test_labels] = load_test_data()
    rp = np.random.permutation(au_train_labels.size)
    digs = au_train_images[rp,:]
    labs = au_train_labels[rp]
    digs = digs[0:1000, :]
    labs = labs[0:1000]
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-lin', action='store_true', default=False)
    parser.add_argument('-poly2', action='store_true', default=False)
    parser.add_argument('-poly3', action='store_true', default=False)    
    parser.add_argument('-rbf', action='store_true', default=False)
    args = parser.parse_args()
    if args.lin:
        '''print('running linear svm')
        ### YOUR CODE HERE
        model = train('linear', au_train_images, au_train_labels)
        print("Computing in-sample and test accuracies...\t (This might also take a few minutes!)")
        in_sample_accuracy = (model.predict(au_train_images)==au_train_labels).mean()
        print('In-sample Acc:\t {:.2%}'.format(in_sample_accuracy))
        test_accuracy = (model.predict(au_test_images)==au_test_labels).mean()
        print('Test Acc:\t {:.2%}'.format(test_accuracy))'''
        cv = svc_cv_lin(digs, labs)

        dataframe = pd.DataFrame(cv) 
        relevant = dataframe.filter(['mean_test_score', 'mean_train_score', 'std_test_score', 'std_train_score', 'param_C', 'mean_fit_time']).sort_values(['mean_test_score'])
        display(relevant)

        # Save the data to a file, then load it again and print it. 
        filename = 'results/svc_lin.csv'
        relevant.to_csv(filename, index=False)
        df = pd.read_csv(filename)
        print('display it again after loading it')
        display(df)               
    if args.poly2:
        print('running poly 2 svm')
        cv = svc_cv_poly(digs, labs, 2)
        # Create a DataFrame object using the cross validation result and filter out the 
        # relevant information. Finally display/print it. 
        dataframe = pd.DataFrame(cv) 
        relevant = dataframe.filter(['mean_test_score', 'mean_train_score', 'std_test_score', 'std_train_score', 'param_C', 'param_coef0', 'mean_fit_time']).sort_values(['mean_test_score'])
        display(relevant)

        # Save the data to a file, then load it again and print it. 
        filename = 'results/svc_poly2.csv'
        relevant.to_csv(filename, index=False)
        df = pd.read_csv(filename)
    if args.poly3:
        print('running poly 3 svm')
        cv = svc_cv_poly(digs, labs, 3)
        # Create a DataFrame object using the cross validation result and filter out the 
        # relevant information. Finally display/print it. 
        dataframe = pd.DataFrame(cv) 
        relevant = dataframe.filter(['mean_test_score', 'mean_train_score', 'std_test_score', 'std_train_score', 'param_C', 'param_coef0', 'mean_fit_time']).sort_values(['mean_test_score'])
        display(relevant)

        # Save the data to a file, then load it again and print it. 
        filename = 'results/svc_poly3.csv'
        relevant.to_csv(filename, index=False)
        df = pd.read_csv(filename)
    if args.rbf:
        print('running rbf svm')
        cv = svc_cv(digs, labs)
        # Create a DataFrame object using the cross validation result and filter out the 
        # relevant information. Finally display/print it. 
        dataframe = pd.DataFrame(cv) 
        relevant = dataframe.filter(['mean_test_score', 'mean_train_score', 'std_test_score', 'std_train_score', 'param_C', 'param_gamma', 'mean_fit_time']).sort_values(['mean_test_score'])
        display(relevant)

        # Save the data to a file, then load it again and print it. 
        filename = 'results/svc_rbf.csv'
        relevant.to_csv(filename, index=False)
        df = pd.read_csv(filename)
        print('display it again after loading it')
        display(df)
        


