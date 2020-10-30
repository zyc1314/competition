import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.utils.data

TRAINDATA_DIR = './train/'
TESTDATA_PATH = './test/testing-X.pkl'
ATTACK_TYPES = {
    'snmp': 0,
    'portmap': 1,
    'syn': 2,
    'dns': 3,
    'ssdp': 4,
    'webddos': 5,
    'mssql': 6,
    'tftp': 7,
    'ntp': 8,
    'udplag': 9,
    'ldap': 10,
    'netbios': 11,
    'udp': 12,
    'benign': 13,
}


class CompDataset(object):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

        self._data = [(x, y) for x, y in zip(X, Y)]

    def __getitem__(self, idx):
        return self._data[idx]

    def __len__(self):
        return len(self._data)


def extract_features(data, has_label=True):

    data['SimillarHTTP'] = 0.
    if has_label:
        return data.iloc[:, -80:-1]

    return data.iloc[:, -79:]


class UserRoundData(object):
    def __init__(self):
        self.data_dir = TRAINDATA_DIR
        self._user_datasets = []
        self.attack_types = ATTACK_TYPES
        self._load_data()

    def _get_data(self, fpath):
        if not fpath.endswith('csv'):
            return

        print('Load User Data: ', os.path.basename(fpath))
        data = pd.read_csv(fpath, skipinitialspace=True, low_memory=False)
        x = extract_features(data)
        y = np.array([
            self.attack_types[t.split('_')[-1].replace('-', '').lower()]
            for t in data.iloc[:, -1]
        ])

        x = x.to_numpy().astype(np.float32)
        x[x == np.inf] = 1.
        x[np.isnan(x)] = 0.
        return (
            x,
            y,
        )

    def _load_data(self):
        _user_datasets = []
        self._user_datasets = []
        for root, dirs, fnames in os.walk(self.data_dir):
            for fname in fnames:
                # each file is for each user
                # user data can not be shared among users
                data = self._get_data(os.path.join(root, fname))
                if data is not None:
                    _user_datasets.append(data)

        for x, y in _user_datasets:
            self._user_datasets.append((
                x,
                y,
            ))

        self.n_users = len(_user_datasets)

    def round_data(self, user_idx, n_round, n_round_samples=-1):
        """Generate data for user of user_idx at round n_round.

        Args:
            user_idx: int,  in [0, self.n_users)
            n_round: int, round number
        """
        if n_round_samples == -1:
            return self._user_datasets[user_idx]

        n_samples = len(self._user_datasets[user_idx][1])
        choices = np.random.choice(n_samples, min(n_samples, n_round_samples))

        return self._user_datasets[user_idx][0][choices], self._user_datasets[
            user_idx][1][choices]

    def uniform_random_loader(self, n_samples, batch_size=1000):
        X, Y = [], []
        n_samples_each_user = n_samples // len(self._user_datasets)
        if n_samples_each_user <= 0:
            n_samples_each_user = 1

        for idx in range(len(self._user_datasets)):
            x, y = self.round_data(user_idx=idx,
                                   n_round=0,
                                   n_round_samples=n_samples_each_user)
            X.append(x)
            Y.append(y)

        data = CompDataset(X=np.concatenate(X), Y=np.concatenate(Y))
        train_loader = torch.utils.data.DataLoader(
            data,
            batch_size=min(batch_size, n_samples),
            shuffle=True,
        )

        return train_loader


def get_test_loader(batch_size=1000):
    with open(TESTDATA_PATH, 'rb') as fin:
        data = pickle.load(fin)

    test_loader = torch.utils.data.DataLoader(
        data['X'],
        batch_size=batch_size,
        shuffle=False,
    )

    return test_loader
