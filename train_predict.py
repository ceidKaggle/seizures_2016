"""train model, using best obtained hyperparameters
"""

from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.grid_search import GridSearchCV
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier

from create_dataset import FEATURES


scale = True                # set that to true to perform dataset scaling
feature_importance = False  # set that to true to plot feature importances
balance = False             # set that to true to balance dataset by randomly
                            # undersampling class with more examples

ds = pd.read_csv('training_dataset.csv')
ds.dropna(inplace=True)  # drop missing values

n_class_0 = len(ds[ds['label'] == 0])
n_class_1 = len(ds[ds['label'] == 1])

print('Dataset loaded')
print('{} examples of class 0 and {} examples of class 1'.format(
                n_class_0, n_class_1))

X_train = ds[FEATURES]
y_train = ds['label']


########################### Data Preprocessing ###############################

if balance:
    print('Balancing Dataset')
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


##################### HYPERPARAMETER TUNING & TRAINING #######################
tuning_parameters = {
        'max_depth': [4, 5, 8, 9, 20, 25, 35],
        'min_child_weight': [1, 2, 3, 4, 5, 6, 8, 10],
        }


print('\n# Tuning hyperparameters')

clf = GridSearchCV(XGBClassifier(), tuning_parameters, cv=5)
clf.fit(X_train, y_train)

print('\nBest parameters set found:')
print(clf.best_params_)

for params, mean_score, scores in clf.grid_scores_:
    print('{} (+/-{}) for {}'.format(
            round(mean_score, 3), 
            round(scores.std() * 2, 3),
            params))
