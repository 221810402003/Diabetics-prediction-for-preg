#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data=pd.read_csv("diabetes.csv")


# In[3]:


data.shape


# In[14]:


data.head(10)


# In[5]:


data.isnull().values.any() #to check for null values


# In[6]:


import seaborn as sns
cm=data.corr()
tcf=cm.index
plt.figure(figsize=(20,20))
g=sns.heatmap(data[tcf].corr(),annot=True,cmap="RdYlGn") #for data visulization we used heatmap


# In[7]:


data.corr() #without the visual graph


# In[8]:


datatrue=len(data.loc[data['Outcome']==True])
datafalse=len(data.loc[data['Outcome']==False]) #to check the ratio of true and flase or +ve and -ve data


# In[9]:


(datatrue,datafalse)


# In[10]:


from sklearn.model_selection import train_test_split
features=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
predicted=['Outcome']
x=data[features].values
y=data[predicted].values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=10)


# In[11]:


from sklearn.impute import SimpleImputer
fill=SimpleImputer(missing_values=0,strategy="mean")
x_train=fill.fit_transform(x_train)
x_test=fill.fit_transform(x_test)


# In[12]:


from sklearn.ensemble import RandomForestClassifier
random_model=RandomForestClassifier(random_state=10)
random_model.fit(x_train,y_train.ravel())


# In[15]:


predictdata=random_model.predict(x_test)
from sklearn import metrics
print("Accuracy={0:.3f}".format(metrics.accuracy_score(y_test,predictdata)))

