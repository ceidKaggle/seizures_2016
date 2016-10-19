from __future__ import division
from sklearn.linear_model import Lasso
import pandas as pd
import numpy as np
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
import matplotlib.pyplot as plt
from imblearn.over_sampling import ADASYN
from sklearn.linear_model import LogisticRegression as LR
from sklearn.preprocessing import RobustScaler

from sklearn.ensemble import RandomForestClassifier as RF

from common import PREDICTORS as FEATURES


scaler = RobustScaler()

FEATURES = [i for i in FEATURES if i != 'patient_id']
results=[]
for i in range(1, 4):
    train_ds = pd.read_csv('training_dataset_2016-10-17_00:28:41.csv')
    train_ds = train_ds[train_ds['patient_id']==i]

    n_class_0 = len(train_ds[train_ds['Class'] == 0])
    n_class_1 = len(train_ds[train_ds['Class'] == 1])

    tmp = train_ds[train_ds['Class']==0].sample(n=round(2*n_class_1), axis=0)
    tmp = tmp.append(train_ds[train_ds['Class']==1], ignore_index=True)
    train_ds = tmp.reset_index()

    print('{} examples of class 0 and {} examples of class 1'.format(
               len(train_ds[train_ds['Class'] == 0]),
               len(train_ds[train_ds['Class'] == 1])))

    X_train = train_ds[FEATURES]
    y_train = train_ds['Class']

    #X_train = scaler.fit_transform(X_train)

    params = {'criterion': ['entropy'],
              'n_estimators': [500, 700],
              'max_depth': [10, 15],
              'max_features': ['sqrt', 'log2'],
              'min_samples_split': [2, 4],
              'min_samples_leaf': [2, 3]
            }
    clf = GridSearchCV(RF(), n_jobs=8, param_grid=params, cv=3)#, iid=False)
    clf.fit(X_train, y_train)

    print clf.best_params_, clf.best_score_

    test_ds = pd.read_csv('testing_dataset_2016-10-17_00:28:41.csv')
    test_ds = test_ds[test_ds['patient_id']==i]
    X_test = test_ds[FEATURES]
    #X_test = scaler.transform(X_test)
    results.append(list(clf.predict_proba(X_test)[:, 1]))


#prediction = [i for i in list(np.mean(np.array(results).T, axis=1))]
test_ds = pd.read_csv('testing_dataset_2016-10-17_00:28:41.csv')
test_ds['Class'][test_ds['patient_id']==1] = results[0]
test_ds['Class'][test_ds['patient_id']==2] = results[1]
test_ds['Class'][test_ds['patient_id']==3] = results[2]
test_ds.to_csv('submission_rf_2.csv', columns=['File', 'Class'], index=False)
