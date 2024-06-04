#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


# In[7]:


df=pd.read_csv(r"E:\online job\DreamWithData\Project List-20240513T051955Z-001\Project List\9. McDonald_s financial statements (2002-2022) Analysis\McDonalds_financial_statements.csv")


# In[8]:


df.head()


# In[9]:


df.describe()


# In[10]:


df.info()


# In[16]:


plt.figure(figsize=(10,5))
sns.heatmap(df.corr(),annot=True)
plt.show()


# In[17]:


sns.pairplot(df)


# In[15]:


plt.plot(df['Year'],df['Market cap ($B)'])
plt.plot(df['Year'],df['Revenue ($B)'])
plt.plot(df['Year'],df['Earnings ($B)'])
plt.xlabel('Year')
plt.ylabel('Value (in $B)')
plt.title('Trend Analysis')
plt.legend()
plt.grid(True)
plt.show()


# In[18]:


plt.figure(figsize=(10, 6))
plt.scatter(df['P/E ratio'], df['Total liabilities ($B)'], color='red')
plt.xlabel('P/E ratio')
plt.ylabel('Total liabilities ($B)')
plt.title('Relationship between P/E ratio and Total liabilities ($B)')
plt.grid(True)
plt.show()


# In[23]:


df[['Revenue ($B)','Year']].value_counts().plot.pie(ylabel='Revenue ($B) vs Year',autopct='%.2f%%')


# In[24]:


df[['Year', 'Revenue ($B)', 'Earnings ($B)']].plot(x='Year', kind='bar', figsize=(12, 8))
plt.xlabel('Year')
plt.ylabel('Value (in $B)')
plt.grid(axis='y')
plt.show()


# In[25]:


df[['Year', 'Net assets ($B)','Total assets ($B)']].plot(x='Year', kind='bar', figsize=(12, 8))
plt.xlabel('Year')
plt.ylabel('Value (in $B)')
plt.grid(axis='y')
plt.show()


# In[29]:


plt.plot(df['Year'],df['Total debt ($B)'],marker='o', label='Total debt ($B)')
plt.plot(df['Year'],df['Total liabilities ($B)'],marker='o', label='Total liabilities ($B)')
plt.plot(df['Year'],df['Total assets ($B)'],marker='o', label='Total assets ($B)')
plt.xlabel('Year')
plt.ylabel('Value (in $B)')
plt.title('Trend Analysis')
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:




