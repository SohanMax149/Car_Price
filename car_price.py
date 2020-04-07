#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import scale
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import copy


# In[2]:


X=pd.read_csv(r'C:\ML\Carprice\car_price.csv')


# In[3]:


Y=X.price 
X=X.drop(['price'],axis=1)


# In[4]:


print(X.describe())


# In[5]:


print('Null values in X',X.isnull().sum())
print('Null values in Y',Y.isnull().sum())


# In[6]:


print(X.head())
print(Y.head())


# In[7]:


print('Shape is',X.shape)


# In[8]:


plt.rcParams["figure.figsize"] = [16,9]
sns.set(style="darkgrid")
sns.pairplot(data=X)


# In[9]:


Y.plot()


# In[10]:


correlation_matrix = X.corr().round(2)
sns.heatmap(data=correlation_matrix, annot=True)


# In[11]:


X["doornumber"]=X["doornumber"].replace({"four": 4, "two": 2})
X["cylindernumber"]=X["cylindernumber"].replace({"four": 4, "six": 6, "five":5, "eight":8, "two": 2, "twelve": 12, "three": 3})


# In[12]:


Cat=X.select_dtypes(include=['object']).copy(deep='False')


# In[13]:


print(Cat)


# In[14]:


Cat=Cat.iloc[:, :].apply(pd.Series)
Name=Cat.CarName.copy()


# In[15]:


print(Cat)
print(type(Cat))
print(Name)


# In[16]:


Temp=[]
Temp=Name.str.split(pat=" ",expand=True)
print(Temp)


# In[17]:


Temp=Temp[0]
X.CarName=Temp
Cat.CarName=Temp
print(Cat.CarName)


# In[18]:


print(Cat.CarName)


# In[19]:


X["CarName"] = X["CarName"].replace({"maxda": "mazda", "porcshce": "porsche", "Nissan": "nissan", "vokswagen": "volkswagen", "toyouta": "toyota", "vw": "volkswagen"})


# In[20]:


L=X.copy(deep="False")
L=pd.get_dummies(L, columns=Cat.columns)
print(L.head())
L.shape


# In[21]:


Xs=scale(L)


# In[22]:


print(Xs)


# In[23]:


X_train, X_test, Y_train, Y_test = train_test_split(Xs,Y,test_size=0.3,random_state=42)


# In[24]:


X_train.shape


# In[25]:


X_test.shape


# In[26]:


Coef=LinearRegression()
Coef.fit(X_train,Y_train)


# In[27]:


Y_pred = Coef.predict(X_test)


# In[28]:


print(Y_pred)


# In[29]:


print(X_test)


# In[30]:


print('mse: %.2f'% mean_squared_error(Y_test,Y_pred))


# In[31]:


print('vs:%.2f'%r2_score(Y_test,Y_pred))


# In[ ]:




