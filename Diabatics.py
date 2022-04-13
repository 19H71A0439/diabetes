#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# In[3]:


from sklearn import svm
from sklearn.metrics import accuracy_score


# In[5]:


salim_diabetes_dataset = pd.read_csv(r'C:\Users\saita\Downloads\diabetes.csv')


# In[6]:


salim_diabetes_dataset.head()


# In[7]:


salim_diabetes_dataset.shape


# In[8]:


salim_diabetes_dataset.describe()


# In[9]:


salim_diabetes_dataset['Outcome'].value_counts()


# In[10]:


salim_diabetes_dataset.groupby('Outcome').mean()


# In[11]:


X = salim_diabetes_dataset.drop(columns = 'Outcome', axis=1)
Y = salim_diabetes_dataset['Outcome']


# In[12]:


scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)
print(standardized_data)


# In[13]:


X = standardized_data
Y = salim_diabetes_dataset['Outcome']


# In[14]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)


# In[16]:


classifier = svm.SVC(kernel='linear')


# In[17]:


classifier.fit(X_train, Y_train)


# In[18]:


X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy score of the training data : ', training_data_accuracy)


# In[19]:


X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score of the test data : ', test_data_accuracy)


# In[20]:


# Step 1
input_data = (5,166,72,19,175,25.8,0.587,51)

# Step 2
input_data_as_numpy_array = np.asarray(input_data)

# Step 3
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# Step 4
std_data = scaler.transform(input_data_reshaped)
print(std_data)

#Step 5
prediction = classifier.predict(std_data)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




