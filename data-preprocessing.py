#!/usr/bin/env python
# coding: utf-8

#Importing packages

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler


#Importing dataset

read_dataset_path = "data/data.csv"
save_dataset_path = "data/preprocessed.csv"

dataset = pd.read_csv(read_dataset_path)

#Visualizing dataset
dataset.columns
dataset.head()
dataset.describe()

#Removing columns insignificant for the model

#Town column is text and has no significant value for the model. Dropping it!

dataset["Town"].value_counts()
dataset = dataset.drop(columns = ["Town"])

#Country column has an error as France is not a part of this assignment. 2 rows are regarding France. Removing these rows from dataset and dropping the column

dataset["Country"].value_counts()
dataset.loc[dataset["Country"] == "France"]
dataset = dataset.drop([dataset.index[39], dataset.index[95]])
dataset["Country"].value_counts()
dataset = dataset.drop(columns="Country")


#Manager names are not required and dropped from dataset

dataset["Manager name"].value_counts()

dataset = dataset.drop(columns = ["Manager name"])

#store ids are not required and dropped from dataset

dataset = dataset.drop(columns = ["Store ID"])

dataset.columns
dataset.head()

#Label encoding all binary columns

dataset["Car park"].value_counts()
dataset["Car park"] = dataset["Car park"].replace("Y", "Yes").replace("N", "No").reset_index()["Car park"]
dataset["Car park"].value_counts()

car_park_LE = LabelEncoder()

dataset["Car park"] = car_park_LE.fit_transform(dataset["Car park"])
dataset["Performance"].value_counts()

performance_LE = LabelEncoder()

dataset["Performance"] = performance_LE.fit_transform(dataset["Performance"])

data = dataset.iloc[:, :].values
data.shape
data[0][5]

location_OHE = OneHotEncoder(sparse_output=False)
locations = location_OHE.fit_transform(np.array(dataset["Location"]).reshape(-1, 1))
locations = locations[:, :-1]
locations.shape
locations = locations.reshape(3, 134)

dataset = dataset.drop(columns = ["Location"])

dataset["location0"] = locations[0]
dataset["location1"] = locations[1]
dataset["location2"] = locations[2]

dataset.head()

columns_to_scale = ["Staff", 
                    "Floor Space",
                    "Window", 
                    "Demographic score", 
                    "40min population", 
                    "30 min population", 
                    "20 min population", 
                    "10 min population",
                    "Store age",
                    "Clearance space",
                    "Competition number",
                    "Competition score"]

for column in columns_to_scale:
    scaler = MinMaxScaler()
    dataset[column] = scaler.fit_transform(np.array(dataset[column]).reshape(134, 1))

dataset.head()
dataset.describe()

#Saving preprocessed dataset to disk

performance = dataset["Performance"]
dataset = dataset.drop(columns = ["Performance"])
dataset["performance"] = performance


dataset.to_csv(save_dataset_path, index=False)





