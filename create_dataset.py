"""perform feature extraction, produce training dataset
Script should be placed in the same directory with the train folders
"""
from __future__ import division

import logging
from functools import partial
import os

from scipy.io import loadmat
import numpy as np
import pandas as pd

# set logger for debugging
logger = logging.getLogger('ceid_kaggle')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '[%(asctime)s - %(name)s - %(levelname)s] - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


def _counter():
    c = 1
    while True:
        yield c
        c += 1


def _avg(signals):
    """expects an ndarray containing signals and returns the avg of all
    signals. 
    """
    return signals.mean()


FEATURES = [
        'avg_1',  # average value for electrode 1
        'avg_2',  # average value for electrode 2 etc
        'avg_3',
        'avg_4',
        'avg_5',
        'avg_6',
        'avg_7',
        'avg_8',
        'avg_9',
        'avg_10',
        'avg_11',
        'avg_12',
        'avg_13',
        'avg_14',
        'avg_15',
        'avg_16',
        ]


def feature_extraction(_counter, _size, file_path):
    """method coordinator of feature extraction. All features get produced
    together to avoid multiples reads of the same file
    """
    try:
        m = loadmat(file_path)
    except Exception as e:
        logger.debug(e)
        # argument in range func should match the size of the feature vector
        return [None for _ in range(len(FEATURES))]  # NaN missing val

    # dictionary that includes all the information extracted from mat file.
    # under the key 'data' you can find 16 signals. each one corresponding 
    # to one electrode. Eg to retrieve the signal from the first electrode
    # you can do sig['data'][:, 0], for the second sig['data'][:, 1] etc
    # under the key 'sequence' you can find the sequence number for preictal
    # states. For a full description of the keys check out the data page. 
    sig = {n: m['dataStruct'][n][0, 0] for n in m['dataStruct'].dtype.names}

    feature_vector = [_avg(sig['data'][:, 0]),
                      _avg(sig['data'][:, 1]),
                      _avg(sig['data'][:, 2]),
                      _avg(sig['data'][:, 3]),
                      _avg(sig['data'][:, 4]),
                      _avg(sig['data'][:, 5]),
                      _avg(sig['data'][:, 6]),
                      _avg(sig['data'][:, 7]),
                      _avg(sig['data'][:, 8]),
                      _avg(sig['data'][:, 9]),
                      _avg(sig['data'][:, 10]),
                      _avg(sig['data'][:, 11]),
                      _avg(sig['data'][:, 12]),
                      _avg(sig['data'][:, 13]),
                      _avg(sig['data'][:, 14]),
                      _avg(sig['data'][:, 15]),
                      ]

    completion = round((_counter() / _size) * 100)
    
    m = 10
    if not completion % m:  # print only for 10%, 20%, 30% etc
        logger.info('training dataset: {}% complete'.format(completion))

    return feature_vector


if __name__ == '__main__':
    dataset = pd.DataFrame(columns=['filename', 'label'])
    train_folders = ['train_1']
    files_labels = []

    # extract names and labels of files to put them in the dataset
    for _dir in train_folders:
        files_labels.extend([(_dir + '/' + fn, fn.split('_')[-1].split('.')[0]) \
                                for fn in os.listdir(_dir)])

    dataset['filename'] = np.asarray(files_labels)[:, 0]
    dataset['label'] = np.asarray(files_labels)[:, 1]

    print('Created index dataset. {0} training examples'.format(len(dataset)))
    

    # declare the features here
    for f in FEATURES:
        dataset[f] = 0

    print('features that will be computed')
    print(FEATURES)

    _counter = _counter().next

    feature_extraction = partial(feature_extraction, _counter, len(dataset))

    dataset[FEATURES] = map(
                feature_extraction,
                dataset['filename']
            )

    logger.info('Dataset Created')
    print(dataset)
    dataset.to_csv('training_dataset.csv', index=False)
