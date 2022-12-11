import pandas as pd
import numpy as np
import csv
import graphviz
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, export_graphviz


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
    #         print(df)

    df_object_col = [col for col in df.columns if df[col].dtype.name == 'object']
    df_int_col = [col for col in df.columns if df[col].dtype.name != 'object' and col != 'income']
    target = df["income"]
    dataset = pd.concat([df[df_int_col], pd.get_dummies(df[df_object_col])], axis=1)

    return target, dataset


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

# GridSearchCV类可以用来对分类器的指定参数值进行详尽搜索，这里搜索最佳的决策树的深度

params = {'max_depth' : range(1, 20)}
best_clf = GridSearchCV(DecisionTreeClassifier(criterion = 'entropy', random_state = 20), param_grid = params)
best_clf = best_clf.fit(train_dataset, train_target)
print(best_clf.best_params_)

# 用决策数进行分类，采用‘熵’作为决策基准，决策深度由上步骤得到8，分裂一个节点所需的样本数至少设为5，并保存预测结果。

# clf = DecisionTreeClassifier() score:0.7836742214851667
classes = [' <=50K', ' >50K']
clf = DecisionTreeClassifier(criterion='entropy', max_depth=8, min_samples_split=5)
clf = clf.fit(train_dataset, train_target)
pred = clf.predict(test_dataset)
print(pred)
score = clf.score(test_dataset, test_target)
# pred = clf.predict_proba(test_dataset)
print(score)
# print(np.argmax(pred, axis = 1))

with open('Predict/DecisionTree.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['id', 'result_pred'])
    for i, result in enumerate(pred):
        writer.writerow([i, classes[result]])
