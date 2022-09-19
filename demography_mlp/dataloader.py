import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pickle
import numpy as np

def normalize(df, mean=None, std=None):
    mean = df.mean()
    pickle.dump(mean, open('resources/norm_mean.pkl', 'wb'))
    std = df.std()
    pickle.dump(mean, open('resources/norm_std.pkl', 'wb'))

    with open('resources/norm_mean.pkl', 'rb') as f:
        mean = pickle.load(f)
    with open('resources/norm_std.pkl', 'rb') as f:
        std = pickle.load(f)
    if mean is None:
        mean = df.mean()
        pickle.dump(mean, open('resources/norm_mean.pkl', 'wb'))
    if std is None:
        std = df.std()
        pickle.dump(mean, open('resources/norm_std.pkl', 'wb'))
    return (df - mean) / (std + np.finfo(float).eps)


class MimicDataset():
    def __init__(self, data, label=None, test=False):
        self.labels = None
        self.test = test
        self.data = data.iloc[:, 3:].values
        self.data = normalize(self.data)
        self.labels = data.iloc[:, 2].values
        if not test: #train set/val set
            self.IDs = data.iloc[:, 0].values
        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.float32)

    def __getitem__(self, index):
        features = self.data[index]
        label = self.labels[index]
        if not self.test: #for train set/val set
            id = self.IDs[index]
            return features, label, id
        if self.test:
            return features, label


        else: #for test set(no label)
            return self.data[index]

    def __len__(self):
        return len(self.data)

train_df = pd.read_csv("../data/phenotyping/train_listfile.csv")
val_df = pd.read_csv("../data/phenotyping/val_listfile.csv")
test_df = pd.read_csv("../data/phenotyping/test_listfile.csv")

train_data = MimicDataset(train_df, label='mortality')
val_data = MimicDataset(val_df, label='mortality')
test_data = MimicDataset(test_df, label='mortality', test=True)



train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=128, shuffle=False)
val_loader = DataLoader(val_data, batch_size=128, shuffle=False)
