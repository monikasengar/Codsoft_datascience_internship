#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df = pd.read_csv(r'C:\Users\monik\Downloads\codsoftds\tested.csv')
print(df.head())


# In[2]:


df.describe()


# In[3]:


df.info()


# In[4]:


df.isnull().sum()


# In[5]:


df.shape


# # Visualization

# In[6]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(14,7))
plt.subplot(2,2,1)
sns.boxplot(x='Sex', y = 'Age',data= df)

plt.subplot(2,2,2)
sns.histplot(df['Fare'],color='g')

plt.subplot(2,2,3)
sns.histplot(df['Age'],color='g')

plt.subplot(2,2,4)
sns.countplot(x='Sex', data=df)

plt.tight_layout()
plt.show()


# In[7]:


sns.countplot(x=df['Sex'], hue=df['Survived'])


# In[8]:


import matplotlib.pyplot as plt
numeric_data = df.select_dtypes(include=[int, float])
correlation_matrix = numeric_data.corr()

sns.heatmap(correlation_matrix, annot=True, cmap='YlGnBu', fmt=".2f")
plt.show()


# In[9]:


# Filled the missing value in Age column with the mean value
df['Age'] = df['Age'].fillna(df['Age'].mean())

# Filled the missing value in Fare columns with mean Value
df['Fare'] = df['Fare'].fillna(df['Fare'].mean())

# Filled the missing value in Embarked columns with backfill method
df['Embarked'] = df['Embarked'].fillna(method='backfill')


# In[11]:


# Drop the columns 
titanic = df.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)
titanic.head(2)


# # Model Building

# # Splitting into test and train set

# In[ ]:


from sklearn.preprocessing import CategoricalEncoder
import category_encoders as ce
encoder = ce.OrdinalEncoder(cols=['Survived','Pclass','Sex','Embarked'])
titanic_encoded = encoder.fit_transform(df)
titanic_encoded.head()


# In[13]:


X = titanic_encoded.drop(['Survived'],axis=1)
y = df['Survived']


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.3,random_state= 21)


# In[ ]:


X_train.head()


# In[ ]:


y.head()


# # Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

print("Training set score: {:.2f}".format(lr.score(X_train, y_train)),'\n')
print("Test set score: {:.2f}".format(lr.score(X_test, y_test)),'\n')
print("Accuracy Score: {:0.2f}".format(accuracy_score(y_test,y_pred_lr)),'\n')
print("Confusion Matrix: \n",confusion_matrix(y_test,y_pred_lr),'\n')
print("Classification_Report: \n",classification_report(y_test,y_pred_lr))


# In[ ]:


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred_lr) * 100
print("Logistic Regression Accuracy:  " +str(round(accuracy,2)) + '%')
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test,y_pred_lr))
print('MSE:', metrics.mean_squared_error(y_test,y_pred_lr))
print('RMSE:',np.sqrt(metrics.mean_absolute_error(y_test,y_pred_lr)))


# In[ ]:




