"""train model, using best obtained hyperparameters
"""
from functools import partial
import os

from imblearn.over_sampling import ADASYN
from sklearn.grid_search import GridSearchCV
from scipy.io import loadmat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier

from common import _avg, _counter
from common import feature_extraction, FEATURES


if __name__ == '__main__':
    scale = False                # set that to true to perform dataset scaling
    feature_importance = False  # set that to true to plot feature importances
    balance = True             # set that to true to balance dataset by
                                # randomly undersampling class with more
                                # examples

    ds = pd.read_csv('training_dataset.csv')
    ds.dropna(inplace=True)  # drop missing values

    n_class_0 = len(ds[ds['label'] == 0])
    n_class_1 = len(ds[ds['label'] == 1])

    print('Dataset loaded')
    print('{} examples of class 0 and {} examples of class 1'.format(
                    n_class_0, n_class_1))

    X_train = np.array(ds[FEATURES]).astype(float)
    y_train = np.array(ds['label'])


########################### Data Preprocessing ###############################
    if balance:
       print('Balancing Dataset')
       ada = ADASYN()
       X_train, y_train = ada.fit_sample(X_train, y_train)
       tmp = ds[ds['label']==0].sample(n=n_class_1, axis=0)
       tmp = tmp.append(ds[ds['label']==1], ignore_index=True)
       ds = tmp.reset_index()

       n_class_0 = len(ds[ds['label'] == 0])
       n_class_1 = len(ds[ds['label'] == 1])

       print('{} examples of class 0 and {} examples of class 1'.format(
                       n_class_0, n_class_1))

    if scale:
        scaler = RobustScaler()
        X_train = scaler.fit_transform(X_train)


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


##################### Hyperparameter Tuning & Training #######################
    tuning_parameters = {
            'max_depth': [35, 40],#[3, 4, 5, 8, 9, 20, 25, 35],
            'min_child_weight': [1],# 2, 3, 4, 5, 6, 8, 10],
            }


    print('\n# Tuning hyperparameters')

    clf = GridSearchCV(XGBClassifier(), tuning_parameters, cv=5)
    clf.fit(X_train, y_train)

    print('\nBest parameters set found:')
    print(clf.best_params_)
    print('score: {}'.format(clf.best_score_))


################################# predict ####################################
    print('testing classifier')
        columns = ['File', 'Class']
        columns.extend(FEATURES)
        dataset = pd.DataFrame(columns=columns)
        test_folders = ['test_1', 'test_2', 'test_3']
        files = []
        idx = []

        for _dir in test_folders:
            files.extend([f for f in os.listdir(_dir)])

        files = sorted(files, key=lambda f:
                    (f.split('_')[0], int((f.split('.')[0]).split('_')[1]))
                )
        dataset['File'] = files

        _counter = _counter().next
        feature_extraction = partial(
                    feature_extraction, _counter, 'test', len(dataset)
                )
        dataset[FEATURES] = map(
                    feature_extraction,
                    dataset['File']
                )

    dataset.to_csv('testing_dataset.csv', index=False)
    dataset['Class'] = clf.predict(np.array(dataset[FEATURES]).astype(float))
    dataset.to_csv('submission_1.csv', columns=['File', 'Class'], index=False)

