# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 22:51:54 2020

@author: HoangHieu
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

import os
data = pd.read_csv("../svmBreast/data.csv")
data.head()
data.keys()   # name of all the columns in our data
data.describe()
X = data.iloc[:, 2:-1]
X.shape
Y = data.iloc[:, 1]
Y.shape
Y = [1 if i=='M' else 0 for i in Y]
Y[1:5]
data.tail()
sns.pairplot(data, hue='diagnosis', vars = ["radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean"])
sns.countplot(data['diagnosis'])
sns.scatterplot(x = 'fractal_dimension_se', y = 'concavity_worst', hue = 'diagnosis', data = data)
sns.scatterplot(x = 'area_mean', y = 'smoothness_mean', hue = 'diagnosis', data = data)
plt.figure(figsize=(20, 10))
sns.heatmap(data.corr(), annot= True)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
min_train = X_train.min()
range_train = (X_train-min_train).max() # find biggest difference between min value and any point of dataset
X_train= (X_train - min_train)/range_train
min_test = X_test.min()
range_test = (X_test-min_test).max()
X_test= (X_test - min_test)/range_test
sns.scatterplot(x = 'area_mean', y = 'smoothness_mean', data = X_train)
print(X_train.shape)
print(X_test.shape)
len(y_train)
classifier = SVC()
classifier.fit(X_train, y_train)
y_predict = classifier.predict(X_test)
y_predict
cm = confusion_matrix(y_test, y_predict)
sns.heatmap(cm, annot= True)
param_grid = {'C':[0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel':['rbf']}
grid = GridSearchCV(SVC(), param_grid, verbose= 4, refit=True)
grid.fit(X_train, y_train)
grid.best_params_
optimized_preds = grid.predict(X_test)
cm = confusion_matrix(y_test, optimized_preds)
sns.heatmap(cm, annot= True)
print(classification_report(y_test, y_predict))
