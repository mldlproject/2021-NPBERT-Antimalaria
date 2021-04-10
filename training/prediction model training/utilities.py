from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import roc_auc_score, average_precision_score

def printPerformance(labels, probs):
    predicted_labels = np.round(probs)
    tn, fp, fn, tp = confusion_matrix(labels, predicted_labels).ravel()
    acc = accuracy_score(labels, predicted_labels)
    roc_auc = roc_auc_score(labels, probs)
    pr_auc = average_precision_score(labels, probs)
    mcc = matthews_corrcoef(labels, predicted_labels)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp)
    f1 = 2*precision*sensitivity / (precision + sensitivity)
    # print('Accuracy: ', round(acc, 4))
    # print('AUC-ROC: ', round(roc_auc, 4))
    # print('AUC-PR: ', round(pr_auc, 4))
    # print('MCC: ', round(mcc, 4))
    # print('Sensitivity/Recall: ', round(sensitivity, 4))
    # print('Specificity: ', round(specificity, 4))
    # print('Precision: ', round(precision, 4))
    # print('F1-score: ', round(f1, 4))
    return round(acc, 4), round(roc_auc, 4), round(pr_auc, 4), round(mcc, 4), round(sensitivity, 4), round(specificity, 4), round(precision, 4), round(f1, 4) 