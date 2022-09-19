import numpy
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import json


class MimicDataset():
    def __init__(self, data, is_test=False):
        self.labels = None
        self.is_test = is_test
        self.data = data.iloc[:, 2].values
        self.labels = data.iloc[:, 1].values
        if not is_test: #train set/val set
            self.IDs = data.iloc[:, 0].values

        for i in range(len(self.data)):
            self.data[i] = json.loads(self.data[i])
            self.data[i] = torch.tensor(self.data[i])
        self.labels = torch.tensor(self.labels, dtype=torch.float32)

    def __getitem__(self, index):
        features = self.data[index]
        label = self.labels[index]
        if not self.is_test: #for train set/val set
            id = self.IDs[index]
            return features, label, id
        return features, label



    def __len__(self):
        return len(self.data)



train_df = pd.read_csv('embeddings/all_train.csv')
val_df = pd.read_csv('embeddings/all_val.csv')

train_data = MimicDataset(train_df)
val_data = MimicDataset(val_df)


train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
val_loader = DataLoader(val_data, batch_size=128, shuffle=False)
