from __future__ import division

import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
import matplotlib.pyplot as plt
from imblearn.over_sampling import ADASYN

from common import PREDICTORS as FEATURES


FEATURES = [i for i in FEATURES if i != 'patient_id']
ds = pd.read_csv('training_dataset_2016-10-16_17:51:48.csv')


print('{} examples of class 0 and {} examples of class 1'.format(
           len(ds[ds['Class'] == 0]), len(ds[ds['Class'] == 1])))

target = 'Class'
IDcol = 'filename'


def modelfit(alg, dtrain, predictors,useTrainCV=True, cv_folds=20,
             early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values,
                              label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain,
                          num_boost_round=alg.get_params()['n_estimators'],
                          nfold=cv_folds, metrics='auc',
                          early_stopping_rounds=early_stopping_rounds,
                          verbose_eval=True)

        alg.set_params(n_estimators=cvresult.shape[0])
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target], eval_metric='auc')
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
    #Print model report:
    print "\nModel Report"
    print "Accuracy : %.4g" % metrics.accuracy_score(
                                dtrain[target].values, dtrain_predictions
                                )
    print "AUC Score (Train): %f" % metrics.roc_auc_score(
                                dtrain[target], dtrain_predprob
                                )
    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    #plt.show()


#Choose all predictors except target & IDcols
predictors = FEATURES
ratio = float(len(ds[ds['Class']==0])) / len(ds[ds['Class']==1])
xgb1 = XGBClassifier(scale_pos_weight=ratio,
                     learning_rate =0.2, n_estimators=135, max_depth=3,
                     min_child_weight=3, gamma=0.3, subsample=0.9,
                     colsample_bytree=0.9, objective='binary:logistic',
                     nthread=8, seed=27, reg_alpha=1e-5)

modelfit(xgb1, ds, predictors)

dataset = pd.read_csv('testing_dataset_2016-10-16_17:51:48.csv')
dataset['Class'] = xgb1.predict(dataset[predictors])
dataset.to_csv('submission_xgb_new_new.csv', columns=['File', 'Class'], index=False)

#param_test = {
#    'max_depth': [12],
#    'min_child_weight': [3, 4],
#    }
#param_test = {
#    'gamma': [0.3, 0.4, 0.5]
#    }
#param_test = {
#        'subsample':[0.8, 0.9],
#        'colsample_bytree':[0.8, 0.9]
#    }

#param_test = {
#        'reg_alpha':[1e-2, 1e-5, 0.1, 1, 100]
#    }

#grid_search_1 = GridSearchCV(xgb1, param_grid=param_test, scoring='roc_auc', iid=False, cv=10)

#grid_search_1.fit(ds[predictors],ds[target])
#print grid_search_1.grid_scores_, grid_search_1.best_params_, grid_search_1.best_score_


