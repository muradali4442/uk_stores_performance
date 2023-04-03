#!/usr/bin/env python
# coding: utf-8

# # Importing required packages

# In[14]:


import numpy as np
import pandas as pd


# In[24]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


# # Importing dataset

# In[15]:


dataset_path = "data/preprocessed.csv"


# In[16]:


dataset = pd.read_csv(dataset_path, index_col = False)


# In[17]:


dataset.columns


# In[18]:


data = dataset.iloc[:, :].values


# # Extracting dependant and independant variables

# In[26]:


y = data[:, -1]


# In[27]:


X = data[:, :-1]


# In[28]:


y.shape


# In[29]:


X.shape


# # Performing train test split

# In[32]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[51]:


classifier = LogisticRegression(random_state=42)
classifier.fit(X_train, y_train)


# In[52]:


y_pred = classifier.predict(X_test)


# In[53]:


cm = confusion_matrix(y_test, y_pred)
cm


# In[54]:


((5 + 13) / (5+5+4+13)) * 100


# In[ ]:




