"""common functionality
"""
from __future__ import division
from copy import deepcopy
import logging

from scipy.io import loadmat
from scipy.stats import kurtosis, skew


# set logger for debugging
logger = logging.getLogger('ceid_kaggle')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '[%(asctime)s - %(name)s - %(levelname)s - %(threadName)s] - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


# since the columns of the training dataset may differ from the actual
# features that will be used in the prediction I created two different
# lists. In predictors you should place all features that you want to
# have on the testing dataset. Hence predictors. Although the patient_id
# may not be used as a predictor but you may still want it in the testing
# dataset for making subject-specific training easier
PREDICTORS = ['patient_id',
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
              'var',
              'skew',
              'kurt']

# sequence is the only feature that can be utilized for training but cannot be
# present in the testing dataset
FEATURES = deepcopy(PREDICTORS)
FEATURES.append('sequence')


# define your feature functions here
def _avg(signals):
    """expects a numpy ndarray containing signals and returns the avg of all
    signals. 
    """
    return signals.mean()


def _var(signals):
    return signals.var()


def _skew(signals):
    return skew(signals).mean()


def _kurt(signals):
    return kurtosis(signals).mean()


def counter():
    c = 1
    while True:
        yield c
        c += 1


def feature_extraction(counter, stage, _size, file_name):
    """method coordinator of feature extraction. All features get produced
    together to avoid multiples reads of the same file
    """
    file_path = '{}_{}/{}'.format(stage, file_name[0], file_name)
    try:
        m = loadmat('../' + file_path, verify_compressed_data_integrity=False)
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

    feature_vector = [int(file_name.split('_')[0]),  #patient id
                      _avg(sig['data'][:, 0]),
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
                      _var(sig['data']),
                      _skew(sig['data']),
                      _kurt(sig['data'])]

    if stage == 'train':
        feature_vector.append(sig['sequence'][0][0])

    completion = round((counter() / _size) * 100)
    
    m = 10
    if not completion % m:  # print only for 10%, 20%, 30% etc
        logger.info('{}ing dataset: {}% complete'.format(stage, completion))

    return feature_vector
