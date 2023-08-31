# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 11:55:47 2023

@author: grago
"""
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


def metrics_score(actual, predicted):
    print(classification_report(actual, predicted))
    cm = confusion_matrix(actual, predicted)
    plt.figure(figsize=(8, 5))
    sns.heatmap(cm, annot=True,  fmt='.2f', xticklabels=['False', 'True'],
                yticklabels=['False', 'True'], cbar=False)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
