from __future__ import division

import pandas as pd
import numpy as np
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
import matplotlib.pyplot as plt
from imblearn.over_sampling import ADASYN
from sklearn.linear_model import LogisticRegression as LR
from sklearn.preprocessing import RobustScaler


from common import PREDICTORS as FEATURES


scaler = RobustScaler()

FEATURES = [i for i in FEATURES if i != 'patient_id']
results=[]
for i in range(11):
    train_ds = pd.read_csv('training_dataset_2016-10-16_19:46:49.csv')
    n_class_0 = len(train_ds[train_ds['Class'] == 0])
    n_class_1 = len(train_ds[train_ds['Class'] == 1])
    tmp = train_ds[train_ds['Class']==0].sample(n=n_class_1, axis=0)
    tmp = tmp.append(train_ds[train_ds['Class']==1], ignore_index=True)
    train_ds = tmp.reset_index()

    print('{} examples of class 0 and {} examples of class 1'.format(
               len(train_ds[train_ds['Class'] == 0]), len(train_ds[train_ds['Class'] == 1])))

    X_train = train_ds[FEATURES]
    y_train = train_ds['Class']

    X_train = scaler.fit_transform(X_train)

    param = {'tol': [1e-1, 1e-3, 1e-5, 1e-7],
             'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
             'penalty': ['l1', 'l2']}

    clf = GridSearchCV(LR(n_jobs=8), param_grid=param, scoring='roc_auc', iid=False, cv=10)
    clf.fit(X_train, y_train)
    print clf.best_score_

    test_ds = pd.read_csv('testing_dataset_2016-10-16_19:46:49.csv')
    X_test = test_ds[FEATURES]
    X_test = scaler.transform(X_test)
    results.append(list(clf.predict(X_test)))

prediction = [int(i) for i in list(np.median(np.array(results).T, axis=1))]
test_ds['Class'] = prediction
test_ds.to_csv('submission_lr_10.csv', columns=['File', 'Class'], index=False)
