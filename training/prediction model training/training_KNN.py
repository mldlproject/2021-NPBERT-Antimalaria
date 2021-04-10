# Import Library
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import roc_auc_score, average_precision_score

import random
from imblearn.pipeline import Pipeline, make_pipeline
from imblearn.over_sampling import SMOTE
from utilities import printPerformance

#--------------------------------------------------------------------#
# Remove Low-variance Features
from sklearn.feature_selection import VarianceThreshold
threshold = (.95 * (1 - .95))

# Normalize Features
from sklearn.preprocessing import MinMaxScaler

#--------------------------------------------------------------------#
# TRAINING MODEL                                                     #
#--------------------------------------------------------------------#
from os import listdir
excludeFiles = ['label.npy', 'RDkit_MD.csv', 'antimalarial_comp_list.csv']

input_PATH = './data'

PATH_list = []
for f in listdir(input_PATH):
    if f not in excludeFiles:
        PATH_list.append(f)

random_seed_list = range(0,1)

# Training Loop
for random_seed in random_seed_list:
    df1 = pd.DataFrame(['Accuracy', 'AUC-ROC', 'AUC-PR','MCC', 'Sensitivity/Recall', 'Specificity', 'Precision', 'F1-score'], columns= ["Metrics"]) 
    for f in range(len(PATH_list)): 
        path = './data' + '/' + PATH_list[f]
        
        #=====================================#
        # Load data
        dataX = np.load(path)
        dataY = np.load('./data/label.npy')
        
        #=====================================#
        # Random Seed = 1
        random.seed(random_seed)

        test_random_pos_samples_list = random.sample(range(1101, 1828), 100)
        test_random_neg_samples_list = random.sample(range(0, 1101), 100)

        train_random_pos_samples_list = list(set(np.arange(1101, 1828)) - set(test_random_pos_samples_list))
        train_random_neg_samples_list = list(set(np.arange(0, 1101)) - set(test_random_neg_samples_list))

        X_test  = np.concatenate((dataX[test_random_pos_samples_list], dataX[test_random_neg_samples_list]))
        y_test  = np.concatenate((dataY[test_random_pos_samples_list], dataY[test_random_neg_samples_list])).reshape(-1)

        X_train = np.concatenate((dataX[train_random_pos_samples_list], dataX[train_random_neg_samples_list]))
        y_train = np.concatenate((dataY[train_random_pos_samples_list], dataY[train_random_neg_samples_list])).reshape(-1)
            
        #=====================================#
        # Set Up Parameter
        my_n_neighbors = np.arange(3,21)
        
        #=====================================#
        # Selecting NOT Using SMOTE or Using SMOTE (Use only one option at each run)
    
        # NOT Use SMOTE
        # my_classifier = make_pipeline(MinMaxScaler(),
        #                               VarianceThreshold(), 
        #                               KNeighborsClassifier())
        
        # Use SMOTE
        my_classifier = make_pipeline(MinMaxScaler(),
                                      VarianceThreshold(), 
                                      SMOTE(random_state=42), 
                                      KNeighborsClassifier())
        
        #=====================================#
        # GridsearchCV
        my_parameters_grid = {'n_neighbors': my_n_neighbors}
        my_new_parameters_grid = {'kneighborsclassifier__' + key: my_parameters_grid[key] for key in my_parameters_grid}

        grid_cv = GridSearchCV(my_classifier, 
                               my_new_parameters_grid, 
                               scoring='roc_auc',
                               n_jobs=-1, 
                               cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42), 
                               return_train_score=True)

        grid_cv.fit(X_train, y_train)

        #=====================================#
        # Create Regressor uing Best Parameters (Use only one option at each run)
    
        best_n_neighbors = grid_cv.best_params_['kneighborsclassifier__n_neighbors']

        # NOT Use SMOTE
        # my_classifier = make_pipeline(make_pipeline(MinMaxScaler(), 
        #                                             VarianceThreshold(), 
        #                                             KNeighborsClassifier(n_neighbors=best_n_neighbors)))  
        
        # Use SMOTE
        my_classifier = make_pipeline(make_pipeline(MinMaxScaler(), 
                                                    VarianceThreshold(),
                                                    SMOTE(random_state=42), 
                                                    KNeighborsClassifier(n_neighbors=best_n_neighbors)))
        
        #=====================================#
        # Testing
        my_classifier.fit(X_train, y_train)
        y_pred = my_classifier.predict(X_test)
        y_prob = my_classifier.predict_proba(X_test)[::,1]

        #=====================================#
        # Evaluation
        x   = printPerformance(y_test, y_prob)
        df  = pd.DataFrame(x, columns = [PATH_list[f].split('.')[0]])
        df1 = pd.concat([df1, df], axis=1)

    pd.DataFrame.to_csv(df1, './results/knn/results_SMOTE_KNN_seed{}.csv'.format(random_seed), index=False)
