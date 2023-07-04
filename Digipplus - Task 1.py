#!/usr/bin/env python
# coding: utf-8

# # Problem Statement - Write an algorithm to hire the best intern from the dataset:

# # ---------------------------------Importing Libraries------------------------------------------------

# In[1]:


import pandas as pd
import numpy as np


# # --------------------------------------Load DataSet----------------------------------------------------

# In[2]:


data =pd.read_csv(r"C:\Users\bodak\Downloads\Applications_for_Machine_Learning_internship_edited.xlsx - Sheet1.csv")


# In[3]:


# Set the display options to show all rows
pd.set_option('display.max_rows', None)
data


# # --------------------------------Data Preprocessing---------------------------------------------------

# In[4]:


data.info()


# In[5]:


data.shape


# In[6]:


data.isna().sum()


# In[7]:


# Get the list of columns with missing values
columns_with_missing_values= data.columns[data.isnull().any()].tolist()


# In[8]:


columns_with_missing_values


# In[9]:


names = ['Name_' + str(i) for i in range(1, len(data['Name']) + 1)]


# In[10]:


names


# In[11]:


# Fill the blank columns with the generated names
data['Name'] = data['Name'].fillna(pd.Series(names))


# In[12]:


data.head(2)


# In[13]:


# Define the new column names
new_column_names = {
    'Name': 'Name',
    'Python (out of 3)': 'Python(out of 3)',
    'Machine Learning (out of 3)': 'ML(out of 3)',
    'Natural Language Processing (NLP) (out of 3)': 'NLP(out of 3)',
    'Deep Learning (out of 3)': 'DL(out of 3)',
    'Other skills': 'Other Skills',
    'Are you available for 3 months, starting immediately, for a full-time work from home internship?': 'Availability',
    'Degree': 'Graduate Degree',
    'Stream': 'Branch/Stream',
    'Current Year Of Graduation': 'Current Year Of Graduation',
    'Performance_PG': 'PG Score',
    'Performance_UG': 'UG Score',
    'Performance_12': '12th Score',
    'Performance_10': '10th Score'
}


# In[14]:


# Rename the columns
data.rename(columns=new_column_names, inplace=True)


# In[15]:


data.head(2)


# In[16]:


#fill null values from other columns
other_columns_with_null_values = ['PG Score','UG Score', '12th Score', '10th Score']


# In[17]:


# Replace NaN values with '0/10'
data[other_columns_with_null_values] = data[other_columns_with_null_values].fillna('0')


# In[18]:


data.head(2)


# In[19]:


# Define the replacement function
def replace_denominator(value):
    if '/' in value:
        numerator, denominator = value.split('/')
        numerator = float(numerator)
        denominator = float(denominator)
        
        if numerator < 10:
            denominator = 10
        elif numerator > 11:
            denominator = 100
        
        return f'{numerator}/{denominator}'
    else:
        return value


# In[20]:


# Replace the denominator based on conditions
data['PG Score'] = data['PG Score'].apply(replace_denominator)
data['UG Score'] = data['UG Score'].apply(replace_denominator)
data['10th Score'] = data['10th Score'].apply(replace_denominator)
data['12th Score'] = data['12th Score'].apply(replace_denominator)


# In[21]:


data.head(55)


# In[22]:


# Delete rows where 'Other Skills' column has null values
data = data.dropna(subset=['Other Skills'])

# Reset the index after dropping rows
data= data.reset_index(drop=True)

# Update the 'Name' column with serial values
data['Name'] = 'Name_' + (data.index + 1).astype(str)


# In[23]:


# Replace null values in 'Graduate Degree' column with 'Other'
data['Graduate Degree'] = data['Graduate Degree'].fillna('Other')


# In[24]:


# Replace null values in '' column with 'Other'
data['Branch/Stream'] = data['Branch/Stream'].fillna('Other')


# In[25]:


# Set the display option to show all rows
pd.set_option('display.max_rows', None)
data


# In[26]:


data.rename(columns={'Are you available for 3 months, starting immediately, for a full-time work from home internship? ' : 'Availability'},inplace=True)


# In[27]:


data.to_csv('New_clean_file.csv')


# In[28]:


data.isna().sum()


# In[29]:


data.shape


# In[30]:


data.dtypes


# In[31]:


data.dtypes


# In[32]:


data


# In[33]:


# Normalize or standardize numerical features if necessary
data[['Python(out of 3)', 'ML(out of 3)', 'DL(out of 3)']] = \
    (data[['Python(out of 3)','ML(out of 3)','DL(out of 3)']]- data[['Python(out of 3)','ML(out of 3)','DL(out of 3)']].mean()) / data[['Python(out of 3)','ML(out of 3)','DL(out of 3)']].std()


# In[34]:


data


# In[35]:


# Define evaluation criteria and assign weights
criteria = ['Python(out of 3)', 'ML(out of 3)', 'DL(out of 3)']
weights = [1,2.5,3]  # Example weights for each criterion


# In[36]:


# Remove "/10" or "/100" from PG, UG, 10th, and 12th score columns
data['PG Score'] = data['PG Score'].str.replace('/10', '')
data['PG Score'] = data['PG Score'].str.replace('/100', '')
data['UG Score'] = data['UG Score'].str.replace('/10', '')
data['UG Score'] = data['UG Score'].str.replace('/100', '')
data['10th Score']= data['10th Score'].str.replace('/10', '')
data['10th Score'] = data['10th Score'].str.replace('/100', '')
data['12th Score'] = data['12th Score'].str.replace('/10', '')
data['12th Score'] = data['12th Score'].str.replace('/100', '')

# Convert the columns to numeric type
data['PG Score'] = pd.to_numeric(data['PG Score'], errors='coerce')
data['UG Score'] = pd.to_numeric(data['UG Score'], errors='coerce')
data['12th Score'] = pd.to_numeric(data['12th Score'], errors='coerce')
data['10th Score'] = pd.to_numeric(data['10th Score'], errors='coerce')


# In[37]:


# Calculate scores for each intern
data['Score'] = data[criteria].mul(weights).sum(axis=1)

# Filter out interns with zero scores in any category (including PG, UG, 12th, and 10th)
filtered_df = data[(data[criteria] > 0).all(axis=1) & (data[['PG Score', 'UG Score', '10th Score', '12th Score']] != 0).all(axis=1)]

# Select the intern with the highest score from the filtered dataframe
best_intern = filtered_df.loc[filtered_df['Score'].idxmax()]


# In[38]:


best_intern


# # -------------------------------------------Conclusion----------------------------------------------------

# In[39]:


# Write the conclusion
conclusion = f"Conclusion:\nThe best intern for the position has been selected based on the evaluation criteria and scores. The chosen intern is 'Name_234', who holds a M.Sc degree and achieved the highest score of 12. They have demonstrated exceptional proficiency in Python, Machine Learning, and Deep Learning. Additionally, their performance in previous academic years, as well as their availability for a full-time work from home internship for 3 months, starting immediately, further supported their selection. We believe that 'Name_234' possesses the necessary skills and qualifications to excel in this internship role.\n"

print(conclusion)


# # -----------------------------------------------Thank You -----------------------------------------------

# In[ ]:




