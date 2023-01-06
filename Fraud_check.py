# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 20:01:54 2023

@author: Rahul
"""

import pandas as pd
import numpy as np
from sklearn.tree import  DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

df = pd.read_csv("D:\\DS\\books\\ASSIGNMENTS\\Decision Trees\\Fraud_check.csv")
df

df.describe()
df.info()
df.dtypes

#visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Check Correlation amoung parameters
corr = df.corr()
fig, ax = plt.subplots(figsize=(8,8))

# Generate a heatmap
sns.heatmap(corr, cmap = 'magma', annot = True, fmt = ".2f")
plt.xticks(range(len(corr.columns)), corr.columns)

plt.yticks(range(len(corr.columns)), corr.columns)

plt.show()

sns.countplot(df['Undergrad'])

sns.countplot(df['Marital.Status'])
plt.show()

sns.countplot(df['Urban'])

sns.pairplot(df,hue="Taxable.Income")
plt.show()

sns.pairplot(df)
plt.show()

#Encoding
import category_encoders as ce
encoder=ce.OrdinalEncoder(cols=['Undergrad', 'Marital.Status', 'Urban'])
df=encoder.fit_transform(df)
df.head()

#treating those who have taxable_income <= 30000 as "Risky" and others are "Good"
tax = []
for value in df["Taxable.Income"]:
    if value<=30000:
        tax.append("Risky")
    else:
        tax.append("Good")

df["tax"] = tax
df.head()

#splitting the data into x and y
x=df.drop(['Taxable.Income','tax'],axis=1)
y = df["tax"]

#Data Partition
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=(40))

model = DecisionTreeClassifier(criterion = 'entropy',max_depth=5)
model.fit(x_train,y_train) 

y_pred_train = model.predict(x_train)
y_pred_test = model.predict(x_test)

from sklearn.metrics import accuracy_score
ac1 = accuracy_score(y_train,y_pred_train)
print("training score:",ac1.round(2))
ac2 = accuracy_score(y_test,y_pred_test)
print("test score:",ac2.round(2))

plt.figure(figsize=(18,10)) 
tree.plot_tree(model,filled=True)
plt.title('Decision tree using Entropy',fontsize=22)
plt.show()

model1 = DecisionTreeClassifier(criterion = 'gini',max_depth=5)
model1.fit(x_train,y_train)

y_pred_train = model1.predict(x_train)
y_pred_test = model1.predict(x_test)

from sklearn.metrics import accuracy_score
ac1 = accuracy_score(y_train,y_pred_train)
print("training score:",ac1.round(2))
ac2 = accuracy_score(y_test,y_pred_test)
print("test score:",ac2.round(2))

plt.figure(figsize=(18,10)) 
tree.plot_tree(model,filled=True)
plt.title('Decision tree using gini',fontsize=22)
plt.show() 

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(x_train,y_train)
y_pred_train = logreg.predict(x_train)
y_pred_test = logreg.predict(x_test)
from sklearn.metrics import accuracy_score
ac1 = accuracy_score(y_train,y_pred_train)
print("training score:",ac1.round(2))
ac2 = accuracy_score(y_test,y_pred_test)
print("test score:",ac2.round(2))

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(max_depth=6)
rfc.fit(x_train,y_train)
y_pred_train = rfc.predict(x_train)
y_pred_test = rfc.predict(x_test)
from sklearn.metrics import accuracy_score
ac1 = accuracy_score(y_train,y_pred_train)
print("training score:",ac1.round(2))
ac2 = accuracy_score(y_test,y_pred_test)
print("test score:",ac2.round(2))






















