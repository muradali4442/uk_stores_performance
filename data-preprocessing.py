#!/usr/bin/env python
# coding: utf-8

# # Importing packages

# In[5]:


import numpy as np
import pandas as pd


# In[6]:


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler


# # Importing dataset

# In[7]:


read_dataset_path = "data/data.csv"
save_dataset_path = "data/preprocessed.csv"


# In[8]:


dataset = pd.read_csv(read_dataset_path)


# # Visualizing dataset

# In[9]:


dataset.columns


# In[10]:


dataset.head()


# In[11]:


dataset.describe()


# # Removing columns insignificant for the model

# Town column is text and has no significant value for the model. Dropping it!

# In[12]:


dataset["Town"].value_counts()


# In[13]:


dataset = dataset.drop(columns = ["Town"])


# ## Error in dataset

# Country column has an error as France is not a part of this assignment. 2 rows are regarding France. Removing these rows from dataset and dropping the column

# In[14]:


dataset["Country"].value_counts()


# In[15]:


dataset.loc[dataset["Country"] == "France"]


# In[16]:


dataset = dataset.drop([dataset.index[39], dataset.index[95]])


# In[17]:


dataset["Country"].value_counts()


# In[18]:


dataset = dataset.drop(columns="Country")


# Manager names are not required and dropped from dataset

# In[19]:


dataset["Manager name"].value_counts()


# In[20]:


dataset = dataset.drop(columns = ["Manager name"])


# store ids are not required and dropped from dataset

# In[21]:


dataset = dataset.drop(columns = ["Store ID"])


# In[22]:


dataset.columns


# In[23]:


dataset.head()


# # Label encoding all binary columns

# In[24]:


dataset["Car park"].value_counts()


# In[25]:


dataset["Car park"] = dataset["Car park"].replace("Y", "Yes").replace("N", "No").reset_index()["Car park"]


# In[26]:


dataset["Car park"].value_counts()


# In[27]:


car_park_LE = LabelEncoder()


# In[28]:


dataset["Car park"] = car_park_LE.fit_transform(dataset["Car park"])


# In[29]:


dataset["Performance"].value_counts()


# In[30]:


performance_LE = LabelEncoder()


# In[31]:


dataset["Performance"] = performance_LE.fit_transform(dataset["Performance"])


# # One hot encoding all categorical columns

# In[32]:


data = dataset.iloc[:, :].values


# In[33]:


data.shape


# In[34]:


data[0][5]


# In[35]:


location_OHE = OneHotEncoder(sparse_output=False)


# In[36]:


locations = location_OHE.fit_transform(np.array(dataset["Location"]).reshape(-1, 1))


# In[37]:


locations = locations[:, :-1]


# In[38]:


locations.shape


# In[39]:


locations = locations.reshape(3, 134)


# In[40]:


dataset = dataset.drop(columns = ["Location"])


# In[41]:


dataset["location0"] = locations[0]
dataset["location1"] = locations[1]
dataset["location2"] = locations[2]


# In[42]:


dataset.head()


# # Scaling all data

# In[43]:


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


# In[44]:


for column in columns_to_scale:
    scaler = MinMaxScaler()
    dataset[column] = scaler.fit_transform(np.array(dataset[column]).reshape(134, 1))


# In[45]:


dataset.head()


# In[46]:


dataset.describe()


# # Saving preprocessed dataset to disk

# In[49]:


performance = dataset["Performance"]
dataset = dataset.drop(columns = ["Performance"])
dataset["performance"] = performance


# In[51]:


dataset.to_csv(save_dataset_path, index=False)


# In[ ]:




