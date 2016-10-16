"""perform feature extraction, produce training dataset
"""
from __future__ import division

from copy import deepcopy
from datetime import datetime
from functools import partial
from multiprocessing.pool import ThreadPool as Pool
from os import listdir as _dir

import numpy as np
import pandas as pd

from common import counter
from common import feature_extraction
from common import logger


__author__ = 'ceidKaggle: gryllos@ceid.upatras.gr'


def create_dataset(folder):
    # checking last letter of folder name to see if testing or training stage 
    if folder.split('_')[0][-1] == 'n':
        from common import FEATURES as features
        stage = 'train'
    else:
        from common import PREDICTORS as features
        stage = 'test'

    columns = deepcopy(features)
    columns.extend(['File', 'Class'])
    ds = pd.DataFrame(columns=columns)
    
    if stage == 'test':
        files = [f for f in _dir(folder)]
        # sort files to be ready for submission
        files = sorted(files, key=lambda f:
                (f.split('_')[0], int((f.split('.')[0]).split('_')[1])))
        ds['File'] = files
    if stage == 'train':
        # extract names and labels of files to put them in the ds
        file_cl = [(f, f.split('_')[-1].split('.')[0]) for f in _dir(folder)]

        ds['File'] = np.asarray(file_cl)[:, 0]
        ds['Class'] = np.asarray(file_cl)[:, 1]

    logger.info('Creating ds: {} rows'.format(len(ds)))
    _counter = counter().next
    _feature_extraction = partial(
            feature_extraction, _counter, stage, len(ds))

    ds[features] = map(_feature_extraction, ds['File'])
    return ds, folder


if __name__ == '__main__':
    # set those options to enable creation of both testing dataset and
    # training dataset
    test = True
    train = True

    dt = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    train_folders = ['../train_1', '../train_2', '../train_3']
    test_folders = ['../test_1', '../test_2', '../test_3']
    
    data_folders = []
    if train:
        data_folders += train_folders
    if test:
        data_folders += test_folders
    
    # parallel creation of training and testing datasets
    pool = Pool(6)
    results = pool.map(create_dataset, data_folders)
    pool.close()
    pool.join()

    if train:
        tr_results = [ds for ds, folder in results if folder in train_folders]
        train_ds = pd.concat(tr_results)
        logger.info('train_ds Created')
        print(train_ds)
        train_ds = train_ds.iloc[np.random.permutation(len(train_ds))]
        train_ds.dropna(inplace=True)
        train_ds = train_ds[train_ds['avg_4']!=0]  # throwing away zero sig
        train_ds.to_csv('training_dataset_{}.csv'.format(dt), index=False)

    if test:
        te_results = [ds for ds, folder in results if folder in test_folders]
        test_ds = pd.concat(te_results)
        logger.info('test_ds Created')
        print(test_ds)
        test_ds.to_csv('testing_dataset_{}.csv'.format(dt), index=False)
