#!/usr/bin/env python
# coding: utf-8

# 
# # Project: Investigate a Dataset - [No-shw appointment]
# 
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# </ul>

# <a id='intro'></a>
# ## Introduction
# 
# ### Dataset Description 
# 
# This data set contains information
# about 10,000 movies collected from
# The Movie Database (TMDb),
# including user ratings and revenue.
# 
# ● Certain columns, like ‘cast’
# and ‘genres’, contain multiple
# values separated by pipe (|)
# characters.
# 
# ● There are some odd characters
# in the ‘cast’ column. Don’t worry
# about cleaning them. You can
# leave them as is.
# 
# ● The final two columns ending
# with “_adj” show the budget and
# revenue of the associated movie
# in terms of 2010 dollars,
# accounting for inflation over
# time.
# 
# 
# ### Question(s) for Analysis
# What factors are importand to predict show up:

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as snb
get_ipython().run_line_magic('matplotlib', 'inline')


# In[40]:


# Upgrade pandas to use dataframe.explode() function. 
get_ipython().system('pip install --upgrade pandas==0.25.0')


# <a id='wrangling'></a>
# ## Data Wrangling
# 
# Loading our data
# 
# 
# ### General Properties
# 

# In[3]:


# Load your data and print out a few lines. Perform operations to inspect data
#   types and look for instances of missing or possibly errant data.
df = pd.read_csv('Database_No_show_appointments/noshowappointments-kagglev2-may-2016.csv')
df.head(5)


# In[4]:


df.shape


# Our data is consists of 110527 rows and 14 columns

# In[5]:


# Is there duplication
df.duplicated().sum()


# No dumlicated rows
# 

# In[6]:


# Is there missing data
df.info()


# No missing data
# 

# In[7]:


# Show data description 
df.describe()


# * Mean age is 37
# * Max age is 115
# * Min age is -19(will be removed)

# In[9]:


# Identifying row of value of -1
mask = df.query('Age=="-1"')
mask


# There is only one row of wrong value which is -1

# 
# ### Data Cleaning
# 
#  

# In[26]:


# Removing wrong value of -1
df.drop(df.index[99832], axis=0, inplace=True)
df


# In[27]:


df.describe()


# In[28]:


# Removing duplicated IDs with duplicated showing status
df.drop_duplicates(['PatientId', 'No-show'], inplace = True)
df.shape


# In[29]:


# Removing data which will not affect the process of analysis
df.drop(['PatientId', 'AppointmentID', 'ScheduledDay', 'AppointmentDay'], axis = 1, inplace = True)
df.head(5)


# ### Data wrangling summary:
# First we showed the dimension of our CSV file , then check if there was duplication appointment. Then if there was duplicated patients ID and remove them. Then we checked if there was missing data. Then getting some useful statistics such as min, max, measn ..etc. 

# <a id='eda'></a>
# ## Exploratory Data Analysis
# 
# Now we're ready to do some computational statistics and create visualization to our data.
# 
# 
# 

# In[30]:


# General view on data
df.hist(figsize=(15, 12));


# In[50]:


df.rename(columns={'No-show':'No_show'}, inplace=True)
df.rename(columns={'Hipertension':'Hypertension'}, inplace=True)


# In[45]:


show = df.No_show == 'No'
noshow = df.No_show == 'Yes'
df[show].count(), df[noshow].count()


# Number of those who showed at the clinic is about 4 times wthose who didn't

# In[48]:


df[show].mean(), df[noshow].mean()


# We've to check our SMS campaign.

# # Factors on the attendance rate

# ## Does age affect the attendance?

# In[55]:


def attendance(df, col_name, attended, absent):
    plt.figure(figsize=[14.7, 8.27])
    df[col_name][show].hist(alpha = 0.5, bins = 10, color = 'yellow', label = 'show')
    df[col_name][noshow].hist(alpha = 0.5, bins = 10, color = 'red', label = 'noshow')
    plt.legend();
    plt.title("Comparison between those who showed to those who didn't  according to Age")
    plt.xlabel("Age")
    plt.ylabel("Patient Number");
attendance(df, 'Age', show, noshow)


# ages between 0 and 3 are most showed(because of parents care), from 45:55 the least attendance

# ## Does age and chronic diseases affect togther 

# In[62]:


plt.figure(figsize=[14.7, 8.27])
df[show].groupby(['Hypertension', 'Diabetes']).mean()['Age'].plot(kind='bar', color = 'yellow', label = 'show')
df[noshow].groupby(['Hypertension', 'Diabetes']).mean()['Age'].plot(kind='bar', color = 'red', label = 'noshow')
plt.legend();
plt.title("Comparison between those who showed to those who didn't  according to Age and chronic diseases")
plt.xlabel("chronic diseases")
plt.ylabel("Mean Age");


# In[49]:


# Compare show showed to who didn't according to their gender
plt.figure(figsize=[14.7, 8.27])
df.Gender[show].hist(alpha = 0.5, label = 'show')
df.Gender[noshow].hist(alpha = 0.5, label = 'noshow')
plt.legend()
plt.title("Comparison between those who showed to those who didn't  according to Gender")
plt.xlabel("Gender")
plt.ylabel("Patient Number");


# Gender has no clear effect 

# ## Does age and gender affects the attendance?
# 

# In[68]:


plt.figure(figsize=[15, 5])
df[show].groupby('Gender').Age.mean().plot(kind='bar', color = 'yellow', label = 'show')
df[noshow].groupby('Gender').Age.mean().plot(kind='bar', color = 'red', label = 'noshow')
plt.legend();
plt.title("Comparison between those who showed to those who didn't  according to Age and gender")
plt.xlabel("Gender")
plt.ylabel("Mean Age");


# In[69]:


print(df[show].groupby("Gender").Age.mean(), df[noshow].groupby("Gender").Age.mean(),
     df[show].groupby('Gender').Age.median(), df[noshow].groupby("Gender").Age.median())


# No clear correlation between age & gender affecting show rate
# 

# ## Does SMS affect the attendance?

# In[71]:


def attendance(df, col_name, attended, absent):
    plt.figure(figsize=[14.7, 8.27])
    df[col_name][show].hist(alpha = 0.5, bins = 15, color = 'yellow', label = 'show');
    df[col_name][noshow].hist(alpha = 0.5, bins = 15, color = 'red', label = 'noshow');
    plt.legend();
    plt.title("Comparison between those who showed to those who didn't  according to receiving SMS")
    plt.xlabel("SMS")
    plt.ylabel("Patient Number");
attendance(df, 'SMS_received', show, noshow)


# We must check our SMS campaign.

# ## Does neighbourhood affect the attendance?

# In[74]:


plt.figure(figsize=[15, 5])
df.Neighbourhood[show].value_counts().plot(kind='bar', color = 'yellow', label = 'show')
df.Neighbourhood[noshow].value_counts().plot(kind='bar', color = 'red', label = 'noshow')

plt.legend()
plt.title("Comparison between those who showed to those who didn't  according to Neighbourhood")
plt.xlabel("Neighbourhood")
plt.ylabel("Patient number");


# It turns out that Neighbourhood has a great effect on attendance

# In[75]:


plt.figure(figsize=[15, 5])
df[show].groupby('Neighbourhood').SMS_received.mean().plot(kind='bar', color = 'yellow', label = 'show')
df[noshow].groupby('Neighbourhood').SMS_received.mean().plot(kind='bar', color = 'red', label = 'show')

plt.legend()
plt.title("Comparison between those who showed to those who didn't  according to Neighbourhood & SMS")
plt.xlabel("Neighbourhood")
plt.ylabel("Patient number");


# Only in 5 Neighbourhoods SMShas response.

# In[76]:


plt.figure(figsize=[15, 5])
df[show].groupby('Neighbourhood').Age.mean().plot(kind='bar', color = 'yellow', label = 'show')
df[noshow].groupby('Neighbourhood').Age.mean().plot(kind='bar', color = 'red', label = 'show')

plt.legend()
plt.title("Comparison between those who showed to those who didn't  according to Neighbourhood & Age")
plt.xlabel("Neighbourhood")
plt.ylabel("Age");


# According to ages patients attendance differ.
# 

# <a id='conclusions'></a>
# ## Conclusions
# 
# * Number of showing patients from same neighbourhood is affected by SMS in addition to age
# * Neighbourhood plays a great role in attendance 
# * Number of showing patients without receiving an SMS is greater than who receives SMS, which means SMScampaign wasn't effective or it needs to be changed.
# 
# 
# ### Limitations
# 
# Correlation between parameters(enrollement in the welfare program, gender, chronic diseases) aren't clear enough
# 
# 

# In[ ]:





# In[ ]:





# In[78]:


from subprocess import call
call(['python', '-m', 'nbconvert', 'Investigate_a_Dataset.ipynb'])

