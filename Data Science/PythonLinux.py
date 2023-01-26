#!/usr/bin/env python
# coding: utf-8

# In[242]:


import pandas as pd

#Program Memasukan data
df = pd.read_csv("data-stroke.csv")
df.info()


# In[243]:


from sklearn import preprocessing

#merubah tipe data dengan menggunakan preprocessing
le = preprocessing.LabelEncoder()
df['gender'] = le.fit_transform(df['gender']).astype("float")
df['ever_married'] = le.fit_transform(df['ever_married'])
df['work_type'] = le.fit_transform(df['work_type'])
df['Residence_type'] = le.fit_transform(df['Residence_type'])
df['smoking_status'] = le.fit_transform(df['smoking_status'])
df.info()


# In[244]:


df.fillna(df.mean())


# In[245]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=1, metric='manhattan')


# In[246]:


import numpy as np

df=df.replace(np.nan, df.mean())


# In[247]:


x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values
print(x.shape)


# In[248]:


# membagi data training dan data set

from sklearn.model_selection import train_test_split

xtrain, xtes, ytrain, ytes = train_test_split(x, y, test_size=0.2, random_state=42)


# In[249]:


knn.fit(xtrain, ytrain)


# In[250]:


hasil=knn.predict(xtes)


# In[251]:


from sklearn.metrics import accuracy_score

akurasi=accuracy_score(ytes, hasil)
print(akurasi)


# In[ ]:




