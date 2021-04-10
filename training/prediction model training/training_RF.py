import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn import metrics
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
        # Load Data
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
        #Set Up Parameter
        my_n_estimators      = 200
        my_max_depth         = np.arange(2, 10)
        my_max_features      = np.arange(0.2, 0.95, 0.05)
        my_min_samples_split = np.arange(2, 10)
        
        #=====================================#
        # Selecting NOT Using SMOTE or Using SMOTE (Use only one option at each run)
        
        # NOT Use SMOTE
        # my_classifier = make_pipeline(MinMaxScaler(),
        #                               VarianceThreshold(), 
        #                               RandomForestClassifier(random_state=42, n_estimators=my_n_estimators))
        
        # Use SMOTE
        my_classifier = make_pipeline(MinMaxScaler(),
                                      VarianceThreshold(), 
                                      SMOTE(random_state=42), 
                                      RandomForestClassifier(random_state=42, n_estimators=my_n_estimators))

        #=====================================#
        # GridsearchCV
        # my_parameters_grid = {'max_depth': my_max_depth, 'max_features': my_max_features} 
        # my_parameters_grid = {'max_depth': my_max_depth, 'min_samples_split': my_min_samples_split}
        my_parameters_grid = {'max_depth': my_max_depth, 'max_features': my_max_features, 'min_samples_split': my_min_samples_split}
        
        my_new_parameters_grid = {'randomforestclassifier__' + key: my_parameters_grid[key] for key in my_parameters_grid}
        
        grid_cv = GridSearchCV(my_classifier, 
                               my_new_parameters_grid, 
                               scoring='roc_auc',
                               n_jobs=-1, 
                               cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42), 
                               return_train_score=True)

        grid_cv.fit(X_train, y_train)

        #=====================================#
        # Create Classifier uing Best Parameters (Use only one option at each run)
        best_max_depth         = grid_cv.best_params_['randomforestclassifier__max_depth']
        best_max_features      = grid_cv.best_params_['randomforestclassifier__max_features']
        best_min_samples_split = grid_cv.best_params_['randomforestclassifier__min_samples_split']

        # NOT Use SMOTE
        # my_classifier = make_pipeline(MinMaxScaler(),
        #                               VarianceThreshold(), 
        #                               RandomForestClassifier(random_state      = 42, 
        #                                                      n_estimators      = my_n_estimators, 
        #                                                      max_depth         = best_max_depth,
        #                                                      max_features      = best_max_features,
        #                                                      min_samples_split = best_min_samples_split
        #                                                      ))
        
        # Use SMOTE
        my_classifier = make_pipeline(MinMaxScaler(),
                                      VarianceThreshold(), 
                                      SMOTE(random_state=42), 
                                      RandomForestClassifier(random_state      = 42, 
                                                             n_estimators      = my_n_estimators, 
                                                             max_depth         = best_max_depth,
                                                             max_features      = best_max_features,
                                                             min_samples_split = best_min_samples_split
                                                             ))                                             
        
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

    pd.DataFrame.to_csv(df1, './results/rf/results_SMOTE_RF_seed{}_md_mf_mss.csv'.format(random_seed), index=False)
