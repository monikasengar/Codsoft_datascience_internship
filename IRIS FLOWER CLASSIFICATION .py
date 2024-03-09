#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df = pd.read_csv(r'C:\Users\monik\Downloads\codsoftds\IRIS.csv')
print(df.head())


# In[2]:


df.info()


# In[3]:


df.isnull().sum()


# In[4]:


df.shape


# In[5]:


df.describe()


# In[ ]:


data = df.drop_duplicates(subset ="species",)
data


# In[6]:


df.value_counts("species")


# # Visualization

# In[7]:


import seaborn as sns
import matplotlib.pyplot as plt


sns.countplot(x='species', data=df, )
plt.show()


# In[8]:


import seaborn as sns
import matplotlib.pyplot as plt


sns.scatterplot(x='sepal_length', y='sepal_width',
                hue='species', data=df, )

# Placing Legend outside the Figure
plt.legend(bbox_to_anchor=(1, 1), loc=2)

plt.show()


# In[9]:


import seaborn as sns
import matplotlib.pyplot as plt


sns.scatterplot(x='petal_length', y='petal_width',
                hue='species', data=df, )

# Placing Legend outside the Figure
plt.legend(bbox_to_anchor=(1, 1), loc=2)

plt.show()


# In[10]:


import seaborn as sns
import matplotlib.pyplot as plt


sns.pairplot(data=df,
             hue='species', height=2)


# In[11]:


import seaborn as sns
import matplotlib.pyplot as plt


fig, axes = plt.subplots(2, 2, figsize=(10,10))

axes[0,0].set_title("sepal_length")
axes[0,0].hist(df['sepal_length'], bins=7)

axes[0,1].set_title("sepal_width")
axes[0,1].hist(df['sepal_width'], bins=5);

axes[1,0].set_title("petal_length")
axes[1,0].hist(df['petal_length'], bins=6);

axes[1,1].set_title("petal_width")
axes[1,1].hist(df['petal_width'], bins=6);


# In[12]:


import seaborn as sns
import matplotlib.pyplot as plt

plot = sns.FacetGrid(df, hue="species")
plot.map(sns.distplot, "sepal_length").add_legend()

plot = sns.FacetGrid(df, hue="species")
plot.map(sns.distplot, "sepal_width").add_legend()

plot = sns.FacetGrid(df, hue="species")
plot.map(sns.distplot, "petal_length").add_legend()

plot = sns.FacetGrid(df, hue="species")
plot.map(sns.distplot, "petal_width").add_legend()

plt.show()


# In[13]:


df = pd.read_csv(r'C:\Users\monik\Downloads\codsoftds\IRIS.csv')
df.drop('species', axis= 1, inplace= True)


df.corr(method='pearson')


# In[14]:


import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv(r'C:\Users\monik\Downloads\codsoftds\IRIS.csv')
df.drop('species', axis= 1, inplace= True)
target_df = pd.DataFrame(columns= ['species'], data= df)
df = pd.concat([df, target_df], axis= 1)
sns.heatmap(df.corr(method='pearson').drop(
['species'], axis=1).drop(['species'], axis=0),
			annot = True);

plt.show()


# # Model Building

# In[3]:


import pandas as pd
df = pd.read_csv(r'C:\Users\monik\Downloads\codsoftds\IRIS.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values


# In[4]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, test_size = 0.3, random_state = 100)


# In[5]:


import numpy as np
from sklearn.model_selection import train_test_split

X = df['sepal_length']
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Selecting the regressor and training the model
from sklearn.tree import DecisionTreeRegressor

model=DecisionTreeRegressor(random_state=0)
model.fit(X_train, y_train)


# In[6]:


y_predict = model.predict(X_test)
print(y_predict)


# In[7]:


from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_predict)
accuracy


# In[ ]:




