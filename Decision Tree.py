import pandas as pd
import numpy as np
import csv
import matplotlib

# import plotTree
matplotlib.use('agg')
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import tree
import graphviz
import pydotplus
import os

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'


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
    #         进行 One-Hot Encoding 对齐
    test_dataset = fix_columns(test_dataset, train_dataset.columns)
    columns = train_dataset.columns
    #         print(df["income"])

    train_target, test_target = np.array(train_target), np.array(test_target)
    train_dataset, test_dataset = np.array(train_dataset), np.array(test_dataset)

    return train_dataset, train_target, test_dataset, test_target, columns


train_dataset, train_target, test_dataset, test_target, columns = Adult_data()
print(train_dataset.shape, test_dataset.shape, train_target.shape, test_target.shape)
print("======================================================================================")
print(train_target)
# GridSearchCV类可以用来对分类器的指定参数值进行详尽搜索，这里搜索最佳的决策树的深度

params = {'max_depth': range(1, 20)}
best_clf = GridSearchCV(DecisionTreeClassifier(criterion='entropy', random_state=20), param_grid=params,cv=10)  # 
best_clf = best_clf.fit(train_dataset, train_target)
print(best_clf.best_params_)

# 用决策数进行分类，采用‘熵’作为决策基准，决策深度由上步骤得到8，分裂一个节点所需的样本数至少设为5，并保存预测结果。

# clf = DecisionTreeClassifier() score:0.7836742214851667
classes = [' <=50K', ' >50K']
clf = DecisionTreeClassifier(criterion='entropy', max_depth=8, min_samples_split=5)
clf = clf.fit(train_dataset, train_target)
text_representation = tree.export_text(clf)
plot = tree.export_graphviz(clf)
print("======================================================================================")
print(text_representation)
print("======================================================================================")
pred = clf.predict(test_dataset)

# fig = plt.figure(figsize=(250, 200))
# _ = tree.plot_tree(clf, feature_names=None, max_depth=8,
#                    class_names=text_representation.title(), filled=True)
#
# fig.savefig("decistion_tree.png")
names = ['age', 'workclass', 'education', 'education-num', 'marital-status',
         'occupation', 'relationship', 'race', 'sex',
         'hours-per-week', 'native-country', 'income']
names = np.array(names, dtype='str')
feature = ["age","Private", "Self-emp-not-inc", "Self-emp-inc""""""", "Federal-gov", "Local-gov", "State-gov", 'Without-pay',
           'Never-worked','Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th',
           '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool',
           'Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse',
           'Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct',
           'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces',
           'Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried',
           'White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black',
           'Female', 'Male','United-States', 'Cambodia', 'England', 'Puerto-Rico','Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India',
           'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica','Vietnam', 'Mexico',
           'Portugal', 'Ireland','France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala',
           'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands','>50K', '<=50K']
features = np.array(feature, dtype='str')
print(clf)
dot_data = tree.export_graphviz(clf, out_file=None, feature_names=features,
                                class_names=names, filled=True, rounded=True,
                                special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("Decision Tree.pdf")
print(train_dataset)
print("======================================================================================")
score = clf.score(test_dataset, test_target)
# pred = clf.predict_proba(test_dataset)
print("The accuracy of Decision Tree Model is %f" % score)  # calculate the maximum depth

# print(np.argmax(pred, axis = 1))

# *************************************************************************************************
# https://blog.csdn.net/wzmsltw/article/details/51057311


with open('Predict/DecisionTree.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['id', 'result_pred'])
    for i, result in enumerate(pred):
        writer.writerow([i, classes[result]])
