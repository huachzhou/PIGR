# -*- coding: utf-8 -*-
"""
create on 18 Sep, 2019

@author: wangshuo

Reference: https://github.com/lijingsdu/sessionRec_NARM/blob/master/data_process.py
"""

import pickle
import torch
from torch.utils.data import Dataset
import numpy as np


def load_data(root, valid_portion=0.1, maxlen=19, sort_by_len=False):
    '''Loads the dataset

    :type path: String
    :param path: The path to the dataset (here RSC2015)
    :type n_items: int
    :param n_items: The number of items.
    :type valid_portion: float
    :param valid_portion: The proportion of the full train set used for
        the validation set.
    :type maxlen: None or positive int
    :param maxlen: the max sequence length we use in the train/valid set.
    :type sort_by_len: bool
    :name sort_by_len: Sort by the sequence lenght for the train,
        valid and test set. This allow faster execution as it cause
        less padding per minibatch. Another mechanism must be used to
        shuffle the train set at each epoch.

    '''

    # Load the dataset
    path_train_data = root + 'train.txt'
    path_test_data = root + 'test.txt'
    path_side_info_data = root + 'side_info.txt'
    with open(path_train_data, 'rb') as f1:
        train_set = pickle.load(f1)

    with open(path_test_data, 'rb') as f2:
        test_set = pickle.load(f2)

    with open(path_side_info_data, 'rb') as f3:
        side_info_set = pickle.load(f3)

    return train_set, test_set, side_info_set


class RecSysDataset(Dataset):
    """define the pytorch Dataset class for yoochoose and diginetica datasets.
    """
    def __init__(self, data):
        self.data = data
        print('-'*50)
        print('Dataset info:')
        print('Number of sessions: {}'.format(len(data[0])))
        print('-'*50)
        
    def __getitem__(self, index):
        session_items = self.data[0][index]
        target_item = self.data[1][index]
        return session_items, target_item

    def __len__(self):
        return len(self.data[0])