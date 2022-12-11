import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import csv
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from sklearn import svm


def add_missing_columns(d, columns):
    missing_col = set(columns) - set(d.columns)
    for col in missing_col:
        d[col] = 0


def fix_columns(d, columns):
    add_missing_columns(d, columns)
    assert (set(columns) - set(d.columns) == set())
    d = d[columns]
    return d


def data_process(df, model):
    df.replace(" ?", pd.NaT, inplace=True)
    if model == 'train':
        df.replace(" >50K", 1, inplace=True)
        df.replace(" <=50K", 0, inplace=True)
    if model == 'test':
        df.replace(" >50K.", 1, inplace=True)
        df.replace(" <=50K.", 0, inplace=True)

    trans = {'workclass': df['workclass'].mode()[0], 'occupation': df['occupation'].mode()[0],
             'native-country': df['native-country'].mode()[0]}
    df.fillna(trans, inplace=True)
    df.drop('fnlwgt', axis=1, inplace=True)
    df.drop('capital-gain', axis=1, inplace=True)
    df.drop('capital-loss', axis=1, inplace=True)

    df_object_col = [col for col in df.columns if df[col].dtype.name == 'object']
    df_int_col = [col for col in df.columns if df[col].dtype.name != 'object' and col != 'income']
    target = df["income"]
    dataset = pd.concat([df[df_int_col], pd.get_dummies(df[df_object_col])], axis=1)

    return target, dataset


class Adult_data(Dataset):
    def __init__(self, model):
        super(Adult_data, self).__init__()
        self.model = model

        df_train = pd.read_csv('adult.csv', header=None,
                               names=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                                      'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                                      'hours-per-week', 'native-country', 'income'])
        df_test = pd.read_csv('adult.test', header=None, skiprows=1,
                              names=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                                     'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                                     'hours-per-week', 'native-country', 'income'])

        train_target, train_dataset = data_process(df_train, 'train')
        test_target, test_dataset = data_process(df_test, 'test')

        #         进行独热编码对齐
        test_dataset = fix_columns(test_dataset, train_dataset.columns)
        #         print(df["income"])
        train_dataset = train_dataset.apply(lambda x: (x - x.mean()) / x.std())
        test_dataset = test_dataset.apply(lambda x: (x - x.mean()) / x.std())
        #         print(train_dataset['native-country_ Holand-Netherlands'])

        train_target, test_target = np.array(train_target), np.array(test_target)
        train_dataset, test_dataset = np.array(train_dataset, dtype=np.float32), np.array(test_dataset,
                                                                                          dtype=np.float32)
        if model == 'test':
            isnan = np.isnan(test_dataset)
            test_dataset[np.where(isnan)] = 0.0
        #             print(test_dataset[ : , 75])

        if model == 'test':
            self.target = torch.tensor(test_target, dtype=torch.int64)
            self.dataset = torch.FloatTensor(test_dataset)
        else:
            #           前百分之八十的数据作为训练集，其余作为验证集
            if model == 'train':
                self.target = torch.tensor(train_target, dtype=torch.int64)[: int(len(train_dataset) * 0.8)]
                self.dataset = torch.FloatTensor(train_dataset)[: int(len(train_target) * 0.8)]
            else:
                self.target = torch.tensor(train_target, dtype=torch.int64)[int(len(train_target) * 0.8):]
                self.dataset = torch.FloatTensor(train_dataset)[int(len(train_dataset) * 0.8):]
        print(self.dataset.shape, self.target.dtype)

    def __getitem__(self, item):
        return self.dataset[item], self.target[item]

    def __len__(self):
        return len(self.dataset)


train_dataset = Adult_data(model='train')
val_dataset = Adult_data(model='val')
test_dataset = Adult_data(model='test')

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=False)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, drop_last=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=False)

def Adult_data():
    df_train = pd.read_csv('adult.csv', header=None,
                           names=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                                  'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                                  'hours-per-week', 'native-country', 'income'])
    df_test = pd.read_csv('adult.test', header=None, skiprows=1,
                          names=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                                 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                                 'hours-per-week', 'native-country', 'income'])

    train_target, train_dataset = data_process(df_train, 'train')
    test_target, test_dataset = data_process(df_test, 'test')
    #         进行独热编码对齐
    test_dataset = fix_columns(test_dataset, train_dataset.columns)
    columns = train_dataset.columns
    #         print(df["income"])

    train_target, test_target = np.array(train_target), np.array(test_target)
    train_dataset, test_dataset = np.array(train_dataset), np.array(test_dataset)

    return train_dataset, train_target, test_dataset, test_target, columns


train_dataset, train_target, test_dataset, test_target, columns = Adult_data()
print(train_dataset.shape, test_dataset.shape, train_target.shape, test_target.shape)



classes = [' <=50K', ' >50K']
clf = svm.SVC(kernel='linear')
clf = clf.fit(train_dataset, train_target)
pred = clf.predict(test_dataset)
score = clf.score(test_dataset, test_target)
print(score)
print(pred)

with open('Predict/SupportVectorMachine.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['id', 'result_pred'])
    for i, result in enumerate(pred):
        writer.writerow([i, classes[result]])
