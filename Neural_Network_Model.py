#!/usr/bin/env python
# coding: utf-8

# # Importing required packages

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


# In[3]:


from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Activation, Flatten


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


# In[13]:


optimizer = SGD(learning_rate=0.01)


# In[14]:


model = Sequential()

model.add(Dense(16, input_shape=(X_train.shape[1],), activation="tanh"))
model.add(Dropout(0.1))

model.add(Dense(32, input_shape=(X_train.shape[1],), activation="tanh"))
model.add(Dropout(0.2))

model.add(Dense(64, activation="relu"))
model.add(Dropout(0.3))

model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))

model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer = optimizer,loss="binary_crossentropy",metrics=["accuracy"])


# In[15]:


model.summary()


# In[18]:


history = model.fit(X_train,
                    y_test,
                    epochs = 200,
                    batch_size = 64,
                    validation_split=0.1)


# In[19]:


model.evaluate(X_test, y_test)


# In[20]:


y_pred = model.predict(X_test)


# In[28]:


y_test


# In[29]:


predicted = []
for i in y_pred:
    if i >=0.5:
        predicted.append(1.)
    else:
        predicted.append(0.)


# In[31]:





# In[32]:


cm = confusion_matrix(y_test, np.array(predicted))
cm


# In[ ]:




