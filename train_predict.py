"""train model, using best obtained hyperparameters
"""
from functools import partial
import os

from sklearn.model_selection import GridSearchCV
from scipy.io import loadmat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler

from common import _avg, counter
from common import feature_extraction, FEATURES


if __name__ == '__main__':
    feature_importance = True   # set that to true to plot feature importances
    balance = False              # set that to true to balance dataset by
                                # randomly undersampling class with more
                                # examples

    ds = pd.read_csv('training_dataset_2016-10-17_00:28:41.csv')
    ds.dropna(inplace=True)  # drop missing values

    n_class_0 = len(ds[ds['Class'] == 0])
    n_class_1 = len(ds[ds['Class'] == 1])

    print('Dataset loaded')
    print('{} examples of class 0 and {} examples of class 1'.format(
                 n_class_0, n_class_1))

    FEATURES = [i for i in FEATURES if i != 'patient_id']
    print FEATURES
    X_train = np.array(ds[FEATURES]).astype(float)
    y_train = np.array(ds['Class'])


########################### Data Preprocessing ###############################
    if balance:
       print('Balancing Dataset')
       print(len(X_train))
       print(len(y_train))
       tmp = ds[ds['Class']==0].sample(n=n_class_1, axis=0)
       tmp = tmp.append(ds[ds['Class']==1], ignore_index=True)
       ds = tmp.reset_index()

       n_class_0 = len(ds[ds['Class'] == 0])
       n_class_1 = len(ds[ds['Class'] == 1])

       print('{} examples of class 0 and {} examples of class 1'.format(
                       n_class_0, n_class_1))


########################## feature importance ################################
    if feature_importance:
        rf = RandomForestClassifier(n_estimators=200).fit(X_train, y_train)

        importances = rf.feature_importances_
        std = np.std(
                [tree.feature_importances_ for tree in rf.estimators_], axis=0
            )
        indices = np.argsort(importances)[::-1]

        print('Feature ranking:')
        for f in range(X_train.shape[1]):
            print "%d. feature %d (%f)" % (f + 1, indices[f],
                                           importances[indices[f]])

        # Plot the feature importances
        plt.figure()
        plt.title("Feature importances")
        plt.bar(range(X_train.shape[1]), importances[indices],
               color="r", yerr=std[indices], align="center")
        plt.xticks(range(X_train.shape[1]), indices)
        plt.xlim([-1, X_train.shape[1]])
        plt.show()
