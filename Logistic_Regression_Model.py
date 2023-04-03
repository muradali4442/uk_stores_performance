#!/usr/bin/env python
# coding: utf-8

# # Importing required packages

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


#Importing dataset

dataset_path = "data/preprocessed.csv"
dataset = pd.read_csv(dataset_path, index_col = False)
dataset.columns

data = dataset.iloc[:, :].values

#Extracting dependant and independant variables

y = data[:, -1]
X = data[:, :-1]
y.shape
X.shape

#Performing train test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classifier = LogisticRegression(random_state=42)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
cm

((5 + 13) / (5+5+4+13)) * 100
