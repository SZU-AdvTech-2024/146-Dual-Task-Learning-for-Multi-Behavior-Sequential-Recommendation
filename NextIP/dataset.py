import pandas as pd
import numpy as np
import random
import math
import multiprocessing
import time
from collections import defaultdict

class Dataset(object):
    def __init__(self, file_prefix, is_valid=1):
        self.data = file_prefix
        if self.data == '../data_process/Tmall/processed_data/Tmall':
            self.train = pd.read_csv(file_prefix + "_train.csv", header=None,
                                     names=['user_id', 'item_id', 'timestamp', 'action_type'],
                                     dtype={'user_id': np.int32, 'item_id': np.int32, 'timestamp': np.int32,'action_type': np.int32})
            self.valid = pd.read_csv(file_prefix + "_valid.csv", header=None,
                                     names=['user_id', 'item_id', 'timestamp', 'action_type'],
                                     dtype={'user_id': np.int32, 'item_id': np.int32, 'timestamp': np.int32,'action_type': np.int32})
        elif self.data == '../data_process/UB/processed_data/UB':
            self.train = pd.read_csv(file_prefix + "_train.csv", header=None,
                                     names=['user_id', 'item_id', 'action_type', 'timestamp'],
                                     dtype={'user_id': np.int32, 'item_id': np.int32, 'action_type': np.int32,
                                            'timestamp': np.int32})
            self.valid = pd.read_csv(file_prefix + "_valid.csv", header=None,
                                     names=['user_id', 'item_id', 'action_type', 'timestamp'],
                                     dtype={'user_id': np.int32, 'item_id': np.int32, 'action_type': np.int32,
                                            'timestamp': np.int32})

        if is_valid == 0:
            self.train = pd.concat([self.train, self.valid], axis='index')
            if self.data == '../data_process/Tmall/processed_data/Tmall':
                self.bet_val_test = pd.read_csv(file_prefix + "_between_val_test.csv", header=None,
                                                names=['user_id', 'item_id', 'timestamp', 'action_type'],
                                                dtype={'user_id': np.int32, 'item_id': np.int32, 'timestamp': np.int32,
                                                       'action_type': np.int32})
                self.valid = pd.read_csv(file_prefix + "_test.csv", header=None,
                                         names=['user_id', 'item_id', 'timestamp', 'action_type'],
                                         dtype={'user_id': np.int32, 'item_id': np.int32, 'timestamp': np.int32,
                                                'action_type': np.int32})
            elif self.data == '../data_process/UB/processed_data/UB':
                self.bet_val_test = pd.read_csv(file_prefix + "_between_val_test.csv", header=None,
                                                names=['user_id', 'item_id', 'action_type', 'timestamp'],
                                                dtype={'user_id': np.int32, 'item_id': np.int32, 'action_type': np.int32,
                                                       'timestamp': np.int32})
                self.valid = pd.read_csv(file_prefix + "_test.csv", header=None,
                                         names=['user_id', 'item_id', 'action_type', 'timestamp'],
                                         dtype={'user_id': np.int32, 'item_id': np.int32, 'action_type': np.int32,
                                                'timestamp': np.int32})

            self.train = pd.concat([self.train, self.bet_val_test], axis='index')

        self.candidate = pd.read_csv(file_prefix + "_negative.csv", header=None)

    def fix_length(self, maxlen=50):
        self.maxlen = maxlen
        self.train.sort_values(by=['user_id', 'timestamp'], axis='index', ascending=True, inplace=True, kind='mergesort')
        self.user_maxid = np.max(self.train.user_id.unique())
        self.item_maxid = np.max(self.train.item_id.unique())  # note that item_maxid > item_size, but it doesn't matter
        self.user_set = set(self.train.user_id.unique())
        self.item_set = set(self.train.item_id.unique())
        self.item_list = list(range(1, self.item_maxid + 1))  # index from 1 to maxid

        self.train = self.train.groupby(['user_id']).tail(maxlen + 1)
        print('real item size:', len(self.item_set), 'user_maxid:', self.user_maxid, 'item_maxid:', self.item_maxid)
        self.train_seq = {}
        self.valid_seq = {}
        self.valid_neg_cand = {}
        self.train_behavior_seq = {}
        self.valid_behaiovr_seq = {}
        for u in self.user_set:
            items = self.train[self.train['user_id'] == u].item_id.values
            seq = np.pad(items, (maxlen+1-len(items), 0), 'constant')  # padding from the left side
            behaviors = self.train[self.train['user_id'] == u].action_type.values
            if self.data == '../data_process/Tmall/processed_data/Tmall':
                behavior_seq = np.pad(behaviors, (maxlen + 1 - len(behaviors), 0), 'constant', constant_values=4)
            elif self.data == '../data_process/UB/processed_data/UB':
                behavior_seq = np.pad(behaviors, (maxlen + 1 - len(behaviors), 0), 'constant', constant_values=4)
            else:
                behavior_seq = None
            self.train_seq[u] = list(seq)
            self.train_behavior_seq[u] = list(behavior_seq)
        # create mask that mark 1 when item is padding, purchases, purchased items
        self.train_mask_real_pos_items = defaultdict(list)
        for u in self.user_set:
            purchase_item = set()
            for i, behavior in enumerate(self.train_behavior_seq[u]):
                if behavior == 0:
                    purchase_item.add(self.train_seq[u][i])

            for i, item in enumerate(self.train_seq[u]):

                if item in purchase_item:
                    self.train_mask_real_pos_items[u].append(0.0)
                else:
                    self.train_mask_real_pos_items[u].append(1.0)
        valid_user_list = list(self.valid.user_id.unique())
        for u in valid_user_list:
            if u in self.train_seq.keys():
                seq = self.train_seq[u][1:]
                target_item = self.valid[self.valid['user_id'] == u].item_id.values[0]
                seq.append(target_item)
                self.valid_seq[u] = seq
                behavior_seq = self.train_behavior_seq[u][1:]
                target_behavior = self.valid[self.valid['user_id'] == u].action_type.values[0]
                behavior_seq.append(target_behavior)
                self.valid_behaiovr_seq[u] = behavior_seq

                self.valid_neg_cand[u] = list(self.candidate[self.candidate[0] == u].values[0][1:])
            else:
                continue

    def sample_batch(self, batch_size=128):
        batch_x = []
        batch_xb = []
        batch_yp = []
        batch_yb = []
        batch_yn = []
        batch_cm = []

        batch_uid = random.sample(self.user_set, batch_size)

        for u in batch_uid:
            x = self.train_seq[u][:-1]
            cm = self.train_mask_real_pos_items[u][:-1]

            xb = self.train_behavior_seq[u][:-1]
            yp = self.train_seq[u][1:]
            yb = self.train_behavior_seq[u][1:]
            yn = random.sample(self.item_set.difference(set(self.train_seq[u])), self.maxlen)
            batch_x.append(x)
            batch_cm.append(cm)

            batch_xb.append(xb)
            batch_yp.append(yp)
            batch_yb.append(yb)
            batch_yn.append(yn)

        return batch_uid, batch_x, batch_xb, batch_yp, batch_yn, batch_yb, batch_cm




