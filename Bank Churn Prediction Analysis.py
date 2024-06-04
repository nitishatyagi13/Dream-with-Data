#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


# In[2]:


df=pd.read_csv(r"E:\online job\DreamWithData\Project List-20240513T051955Z-001\Project List\4. Bank Churn Prediction Analysis\BankChurners.csv")


# In[3]:


df.head()


# In[4]:


df["Attrition_Flag"].value_counts()


# In[5]:


del df["Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2"]


# In[6]:


del df["Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1"]


# In[7]:


df.head()


# In[8]:


df.isnull().sum()


# In[9]:


# Import label encoder 
from sklearn import preprocessing 
  
# label_encoder object knows  
# how to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 
  
# Encode labels in column 'species'. 
df['Attrition_Flag']= label_encoder.fit_transform(df['Attrition_Flag']) 
  
df['Attrition_Flag'].value_counts()


# In[10]:


cont=df.select_dtypes(include=np.number)


# In[11]:


cont.describe()


# In[12]:


for i in cont:
    plt.figure(figsize=(10,5))
    sns.boxplot(y=i,data=df)
    plt.show()


# In[13]:


sns.histplot(x='Customer_Age', data=df, )
plt.show()


# In[14]:


plt.figure(figsize=(10,5))
sns.heatmap(df.corr(),annot=True)
plt.show()


# In[15]:


colors = ['#4d2734','#dd8c8c']
Marital_Attrition_Flag = df.groupby(['Marital_Status','Attrition_Flag']).size().unstack()

ax = (Marital_Attrition_Flag.T*100.0 / Marital_Attrition_Flag.T.sum()).T.plot(kind='bar',width = 0.2,stacked = True,rot = 0,figsize = (8,6),color = colors)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.legend(loc='center',prop={'size':14},title = 'Attrition_Flag')
ax.set_ylabel('% Customers')
ax.set_title('Attrition_Flag by Marital_status',size = 14)

# Code to add the data labels on the stacked bar chart
for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy() 
    ax.annotate('{:.0f}%'.format(height), (p.get_x()+.25*width, p.get_y()+.4*height),
                color = 'white',
               weight = 'bold',size =14)


# In[16]:


colors = ['#4D3425','#E4512B']
Card_Attrition_Flag = df.groupby(['Card_Category','Attrition_Flag']).size().unstack()

ax = (Card_Attrition_Flag.T*100.0 / Card_Attrition_Flag.T.sum()).T.plot(kind='bar',width = 0.2,stacked = True,rot = 0,figsize = (8,6),color = colors)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
ax.legend(loc='center',prop={'size':14},title = 'Attrition_Flag')
ax.set_ylabel('% Customers')
ax.set_title('Attrition_Flag by Card_Category',size = 14)

# Code to add the data labels on the stacked bar chart
for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy() 
    ax.annotate('{:.0f}%'.format(height), (p.get_x()+.25*width, p.get_y()+.4*height),
                color = 'white',
               weight = 'bold',size =14)


# In[17]:


#customers are likely to churn when the Total_Trans_Amt is less
ax = sns.kdeplot(df.Total_Trans_Amt[(df["Attrition_Flag"] == 0)],fill=True)
ax = sns.kdeplot(df.Total_Trans_Amt[(df["Attrition_Flag"] == 1)],fill=True)
ax.legend(["Attrited customer","Existing customer"],loc='upper right')
ax.set_ylabel('Density')
ax.set_xlabel('Total_Trans_Amt')
ax.set_title('Distribution of Total_Trans_Amt by Attrition_Flag')


# In[18]:


#customers are likely to churn when the Total_Revolving_Bal is less
ax = sns.kdeplot(df.Total_Revolving_Bal[(df["Attrition_Flag"] == 0)],fill=True)
ax = sns.kdeplot(df.Total_Revolving_Bal[(df["Attrition_Flag"] == 1)],fill=True)
ax.legend(["Attrited customer","Existing customer"],loc='upper right')
ax.set_ylabel('Density')
ax.set_xlabel('Total_Revolving_Bal')
ax.set_title('Distribution of Total_Revolving_Bal by Attrition_Flag')


# In[19]:


sns.kdeplot(df["Credit_Limit"])


# In[20]:


X = df[["Dependent_count", "Credit_Limit","Total_Revolving_Bal","Total_Trans_Amt", "Total_Ct_Chng_Q4_Q1","Months_Inactive_12_mon","Contacts_Count_12_mon"]]
y = df["Attrition_Flag"]


# In[22]:


from sklearn.model_selection import train_test_split


# In[24]:


from sklearn.preprocessing import StandardScaler


# In[25]:


# Split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)
X.describe()
# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test= sc.transform(X_test)


# In[26]:


pip install lazypredict


# In[27]:


from lazypredict.Supervised import LazyClassifier


# In[28]:


clf=LazyClassifier()


# In[29]:


models,predictions=clf.fit(X_train,X_test,y_train,y_test)


# In[30]:


print(models)


# In[31]:


from sklearn.ensemble import RandomForestClassifier


# In[32]:


# creating a RF classifier
clf = RandomForestClassifier(n_estimators = 100) 

# Training the model on the training dataset
# fit function is used to train the model using the training sets as parameters
clf.fit(X_train, y_train)

# performing predictions on the test dataset
y_pred = clf.predict(X_test)

# metrics are used to find accuracy or error
from sklearn import metrics 
print()

# using metrics module for accuracy calculation
print("ACCURACY OF THE MODEL:", metrics.accuracy_score(y_test, y_pred))


# In[34]:


from sklearn.metrics import confusion_matrix,accuracy_score,classification_report,roc_curve


# In[35]:


predictions = clf.predict(X_test)
print(classification_report(y_test, predictions))


# In[ ]:




