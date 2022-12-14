import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
# PS. we need to install sklearn by using command `pip install scikit-learn==0.22`
df = pd.read_csv('adult.csv', header=None,
                 names=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                        'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                        'native-country', 'income'])
df.head()
df.info()  ###

df.apply(lambda x: np.sum(x == ""))

df.replace(" ?", pd.NaT, inplace=True)
df.replace(" >50K", 1, inplace=True)  # substitude >50K by 1
df.replace(" <=50K", 0, inplace=True)  # substitude <=50K by 1
trans = {'workclass': df['workclass'].mode()[0], 'occupation': df['occupation'].mode()[0],
         'native-country': df['native-country'].mode()[0]}
df.fillna(trans, inplace=True)
f = open("a.txt","w")
f.write(str(df.describe()))  ###
f.close()

df.drop('fnlwgt', axis=1, inplace=True)  # cancel the num
df.drop('capital-gain', axis=1, inplace=True)
df.drop('capital-loss', axis=1, inplace=True)
df.head()
# more than 75% cancel capital-gain and capital-loss


import matplotlib.pyplot as plt

plt.scatter(df["income"], df["age"])
plt.grid(b=True, which="major", axis='y')
plt.title("Income distribution by age (1 is >50K)")
plt.show()  # income distribution by age

df["workclass"].value_counts()
# income distribution by work class: for higher age class, > 50K is larger than <= 50K

income_0 = df["workclass"][df["income"] == 0].value_counts()
income_1 = df["workclass"][df["income"] == 1].value_counts()
df1 = pd.DataFrame({" >50K": income_1, " <=50K": income_0})
df1.plot(kind='bar', stacked=True)
plt.title("income distribution by Workclass")
plt.xlabel("workclass")
plt.ylabel("number of person")
plt.show()

df1 = df["hours-per-week"].groupby(df["workclass"]).agg(['mean', 'max', 'min'])
df1.sort_values(by='mean', ascending=False)
print(df1)



income_0 = df["education"][df["income"] == 0].value_counts()
income_1 = df["education"][df["income"] == 1].value_counts()
df1 = pd.DataFrame({" >50K" : income_1, " <=50K" : income_0})
df1.plot(kind = 'bar', stacked = True)
plt.title("income distribution by Workclass")
plt.xlabel("education")
plt.ylabel("number of person")
plt.show()

income_0 = df["education-num"][df["income"] == 0]
income_1 = df["education-num"][df["income"] == 1]
df1 = pd.DataFrame({' >50K' : income_1, ' <=50K' : income_0})
df1.plot(kind = 'kde')
plt.title("education of income")
plt.xlabel("education-num")
plt.show()

# fig, ([[ax1, ax2, ax3], [ax4, ax5, ax6]]) = plt.subplots(2, 3, figsize=(15, 10))
fig = plt.figure(figsize = (15, 10))

ax1 = fig.add_subplot(231)
income_0 = df[df["race"] == ' White']["relationship"][df["income"] == 0].value_counts()
income_1 = df[df["race"] == ' White']["relationship"][df["income"] == 1].value_counts()
df1 = pd.DataFrame({' >50K' : income_1, ' <=50K' : income_0})
df1.plot(kind = 'bar', ax = ax1)
ax1.set_ylabel('number of person')
ax1.set_title('income of relationship by race_White')

ax2 = fig.add_subplot(232)
income_0 = df[df["race"] == ' Black']["relationship"][df["income"] == 0].value_counts()
income_1 = df[df["race"] == ' Black']["relationship"][df["income"] == 1].value_counts()
df2 = pd.DataFrame({' >50K' : income_1, ' <=50K' : income_0})
df2.plot(kind = 'bar', ax = ax2)
ax2.set_ylabel('number of person')
ax2.set_title('income of relationship by race_Black')

ax3 = fig.add_subplot(233)
income_0 = df[df["race"] == ' Asian-Pac-Islander']["relationship"][df["income"] == 0].value_counts()
income_1 = df[df["race"] == ' Asian-Pac-Islander']["relationship"][df["income"] == 1].value_counts()
df3 = pd.DataFrame({' >50K' : income_1, ' <=50K' : income_0})
df3.plot(kind = 'bar', ax = ax3)
ax3.set_ylabel('number of person')
ax3.set_title('income of relationship by race_Asian-Pac-Islander')

ax4 = fig.add_subplot(234)
income_0 = df[df["race"] == ' Amer-Indian-Eskimo']["relationship"][df["income"] == 0].value_counts()
income_1 = df[df["race"] == ' Amer-Indian-Eskimo']["relationship"][df["income"] == 1].value_counts()
df4 = pd.DataFrame({' >50K' : income_1, ' <=50K' : income_0})
df4.plot(kind = 'bar', ax = ax4)
ax4.set_ylabel('number of person')
ax4.set_title('income of relationship by race_Amer-Indian-Eskimo')

ax5 = fig.add_subplot(235)
income_0 = df[df["race"] == ' Other']["relationship"][df["income"] == 0].value_counts()
income_1 = df[df["race"] == ' Other']["relationship"][df["income"] == 1].value_counts()
df5 = pd.DataFrame({' >50K' : income_1, ' <=50K' : income_0})
df5.plot(kind = 'bar', ax = ax5)
ax5.set_ylabel('number of person')
ax5.set_title('income of relationship by race_Other')

plt.tight_layout()
plt.show()



# fig, ([[ax1, ax2, ax3], [ax4, ax5, ax6]]) = plt.subplots(2, 3, figsize=(10, 5))
fig = plt.figure()

ax1 = fig.add_subplot(121)
income_0 = df[df["sex"] == ' Male']["occupation"][df["income"] == 0].value_counts()
income_1 = df[df["sex"] == ' Male']["occupation"][df["income"] == 1].value_counts()
df1 = pd.DataFrame({' >50K' : income_1, ' <=50K' : income_0})
df1.plot(kind = 'bar', ax = ax1)
ax1.set_ylabel('number of person')
ax1.set_title('income of occupation by sex_Male')

ax2 = fig.add_subplot(122)
income_0 = df[df["sex"] == ' Female']["occupation"][df["income"] == 0].value_counts()
income_1 = df[df["sex"] == ' Female']["occupation"][df["income"] == 1].value_counts()
df2 = pd.DataFrame({' >50K' : income_1, ' <=50K' : income_0})
df2.plot(kind = 'bar', ax = ax2)
ax2.set_ylabel('number of person')
ax2.set_title('income of occupation by sex_Female')

plt.tight_layout()
plt.show()







df_object_col = [col for col in df.columns if df[col].dtype.name == 'object']
df_int_col = [col for col in df.columns if df[col].dtype.name != 'object' and col != 'income']
target = df["income"]
dataset = pd.concat([df[df_int_col], pd.get_dummies(df[df_object_col])], axis = 1)
print(dataset.head())



