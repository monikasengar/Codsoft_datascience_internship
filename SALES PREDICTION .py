#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df = pd.read_csv(r'C:\Users\monik\Downloads\codsoftds\advertising.csv')
print(df.head())


# In[2]:


df.info()


# In[3]:


df.isnull().sum()


# In[4]:


df.shape


# In[5]:


df.describe()


# In[6]:


df.memory_usage()


# In[7]:


df.value_counts


# In[8]:


df.index


# In[10]:


import seaborn as sns
import matplotlib.pyplot as plt
plt.scatter(x='TV', y='Sales', data=df)


# # Data cleaning

# In[11]:


import pandas as pd
df = pd.read_csv(r'C:\Users\monik\Downloads\codsoft2\sales_prediction\advertising.csv')
df.isnull().sum()*100/df.shape[0]


# # Outliers

# In[12]:


import matplotlib.pyplot as plt
import seaborn as sns
fig, axs = plt.subplots(3, figsize = (5,5))
plt1 = sns.boxplot(df['TV'], ax = axs[0])
plt2 = sns.boxplot(df['Newspaper'], ax = axs[1])
plt3 = sns.boxplot(df['Radio'], ax = axs[2])
plt.tight_layout()


# # Exploratory  Data Analysis

# In[13]:


sns.boxplot(df['Sales'])
plt.show()


# In[14]:


sns.pairplot(df, x_vars=['TV', 'Newspaper', 'Radio'], y_vars='Sales', height=4, aspect=1, kind='scatter')
plt.show()


# In[15]:


sns.heatmap(df.corr(), cmap="YlGnBu", annot = True)
plt.show()


# # Model Building

# In[16]:


X = df['TV']
y = df['Sales']


# In[17]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, test_size = 0.3, random_state = 100)


# In[18]:


X_train.head()


# In[19]:


y_train.head()


# In[20]:


import statsmodels.api as sm


# In[21]:


# Add a constant to get an intercept
X_train_sm = sm.add_constant(X_train)

# Fit the resgression line using 'OLS'
lr = sm.OLS(y_train, X_train_sm).fit()


# In[22]:


lr.params


# In[23]:


print(lr.summary())


# In[24]:


plt.scatter(X_train, y_train)
plt.plot(X_train, 6.948 + 0.054*X_train, 'r')
plt.show()


# In[25]:


y_train_pred = lr.predict(X_train_sm)
res = (y_train - y_train_pred)


# In[26]:


fig = plt.figure()
sns.distplot(res, bins = 15)
fig.suptitle('Error Terms', fontsize = 15)                  # Plot heading 
plt.xlabel('y_train - y_train_pred', fontsize = 15)         # X-label
plt.show()


# In[27]:


plt.scatter(X_train,res)
plt.show()


# # Model Evaluation

# In[28]:


# Add a constant to X_test
X_test_sm = sm.add_constant(X_test)

# Predict the y values corresponding to X_test_sm
y_pred = lr.predict(X_test_sm)


# In[29]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# In[30]:


r_squared = r2_score(y_test, y_pred)
r_squared

