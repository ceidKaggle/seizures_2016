# seizures_2016
A repo to host our kaggle 2016 seizures prediction competition scripts

## How to use

### for *nix users
`pip install -r requirements.txt` or `sudo pip install -r requirements.txt`

### for windows users
better install a python bundled with all or most of the requirements like `anacondas` and install manually
only the packages that are not already included in the bundle.

After requirements are install use `python create_dataset.py` to create training dataset and `python train_predict.py` to
train model.

Tested only with python2.7 but probably compatible with python 3+ as well.
