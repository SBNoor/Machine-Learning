import os
from zipfile import ZipFile


files = ['cnn_model.py',
         'h2_util.py',
         'model.py',
         'model_stats.py',
         'nn_model.py',
         'svm.py',
         'predict.py',
         'visualize_models.py']

files = ['cnn_model_solved.py',
         'h2_util.py',
         'model.py',
         'model_stats.py',
         'nn_model_solved.py',
         'svm_solved.py',
         'predict.py',
         'visualize_models.py']

results = ['cnn_1024_early_stopping.csv',
           'cnn_1024_confusion_matrix.csv',
           'cnn_1024_early_stopping.png',
           'cnn_1024_stats_accuracy.csv',
           'nn_256_early_stopping.csv',
           'nn_256_confusion_matrix.csv',
           'nn_256_early_stopping.png',
           'nn_256_stats_accuracy.csv']

results = []

# CHANGE your_best_classifier_weight_filename to your file - all files with that prefix in model_weights are included
your_best_classifier_weight_filename = 'nn_h256_THE_BEAST.weight'
### YOUR CODE HERE
### END CODE

           
with ZipFile('handin2_upload_files.zip', 'w') as myzip:
    for filename in files:
        myzip.write(filename)
        print('zipping {0}'.format(filename))
    for filename in results:
        rf = os.path.join('results', filename)
        print('zipping {0}'.format(rf))
        myzip.write(rf)
    model_name = your_best_classifier_weight_filename[0]
    weight_files = os.listdir('model_weights')
    relevant_files = [x for x in weight_files if x.startswith(your_best_classifier_weight_filename)]
    print('Model Weight Files', relevant_files)
    for filename in relevant_files:
        rf = os.path.join('model_weights', filename)
        print('zipping {0}'.format(rf))
        myzip.write(rf)

