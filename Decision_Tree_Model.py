#!/usr/bin/env python
# coding: utf-8

# # Importing required packages

# In[1]:


import numpy as np
import pandas as pd


# In[3]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix


# # Importing dataset

# In[4]:


dataset_path = "data/preprocessed.csv"


# In[5]:


dataset = pd.read_csv(dataset_path, index_col = False)


# In[6]:


dataset.columns


# In[7]:


data = dataset.iloc[:, :].values


# # Extracting dependant and independant variables

# In[8]:


y = data[:, -1]


# In[9]:


X = data[:, :-1]


# In[10]:


y.shape


# In[11]:


X.shape


# # Performing train test split

# In[12]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[14]:


classifier = DecisionTreeClassifier(random_state=42)
classifier.fit(X_train, y_train)


# In[15]:


y_pred = classifier.predict(X_test)


# In[16]:


cm = confusion_matrix(y_test, y_pred)
cm


# In[18]:


((5 + 12) / (5+5+5+12)) * 100


# In[ ]:




