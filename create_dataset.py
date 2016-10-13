"""perform feature extraction, produce training dataset
Script should be placed in the same directory with the train folders
"""
from __future__ import division

from functools import partial
import os

import numpy as np
import pandas as pd

from common import _avg      # import feature functions
from common import feature_extraction
from common import FEATURES  # import feature names


if __name__ == '__main__':
    dataset = pd.DataFrame(columns=['filename', 'label'])
    train_folders = ['train_1', 'train_2', 'train_3']
    files_labels = []

    # extract names and labels of files to put them in the dataset
    for _dir in train_folders:
        files_labels.extend([(fn, fn.split('_')[-1].split('.')[0]) \
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

    feature_extraction = partial(
                feature_extraction, 'train', _counter, len(dataset)
            )

    dataset[FEATURES] = map(
                feature_extraction,
                dataset['filename']
            )

    logger.info('Dataset Created')
    print(dataset)
    dataset.to_csv('training_dataset.csv', index=False)
