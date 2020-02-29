#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


housing = pd.read_csv("dataset.csv")


# In[3]:


housing.head()


# In[4]:


housing.info()


# In[5]:


housing['CHAS'].value_counts()# categorical data


# In[6]:


housing.describe()


# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


# For Ploting Histogram
#import matplotlib.pyplot as plt
#housing.hist(bins = 50, figsize = (20,15))


# ## Train Test Splitting

# In[9]:


# For Learning Purpose
import numpy as np
def split_train_test(data,test_ratio):
    np.random.seed(40)# Fix the permutation
    shuffled = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


# In[10]:


#train_set,test_set = split_train_test(housing, 0.2)


# In[11]:


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size = 0.2, random_state = 42)


# In[12]:


print("Train Length:{train} & test Lenght:{test}".format(train = len(train_set),test = len(test_set)))


# In[13]:


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)
for train_index, test_index in split.split(housing, housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


# In[14]:


strat_test_set['CHAS'].value_counts()


# In[15]:


strat_train_set['CHAS'].value_counts()


# In[16]:


#95/7


# In[17]:


#376/28


# ## Copy training data beacuse we will work on that

# In[18]:


housing = strat_train_set.copy()


# ## Looking for Coorelation
# coorelation lies between 1 to -1
# +1 means: strong positive coorelation: if one number incerase coresponsd number also increase
# -1 means: strong negative coorelation: if one number increase correspond number decrease
# 

# In[19]:


corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending = False)


# In[20]:


corr_matrix['MEDV'].sort_values(ascending = False)


# In[21]:


from pandas.plotting import scatter_matrix
attributes = ['MEDV','RM','ZN','LSTAT']
#scatter_matrix(housing[attributes],figsize = (12,8))


# In[22]:


#housing.plot(kind = 'scatter', x = 'RM',y = 'MEDV', alpha = 0.6)


# In[23]:


#housing['TAXRM'] = housing['TAX'] / housing['RM']


# In[24]:


housing.head()


# In[25]:


corr_matrix = housing.corr() # Check TAXRm coorelation
corr_matrix['MEDV'].sort_values(ascending = False)


# In[26]:


#housing.plot(kind = 'scatter', x = 'TAXRM',y = 'MEDV', alpha = 0.6)


# In[27]:


housing = strat_train_set.drop('MEDV', axis = 1)
housing_labels = strat_train_set['MEDV'].copy()


# In[28]:


housing.shape


# ## Missing Attributes

# In[29]:


# To take care of missing data,you have three option:
#     1. get rid of missing data
#     2. get rid of whole attributes
#     3. fill missing data with 0 mean or median


# In[30]:


# First Option
a = housing.dropna(subset = ['RM'])
a.shape
#a.head()


# In[31]:


# Second Option, drop if this column has week coorelation
b = housing.drop('RM', axis = 1)
b.head()


# In[32]:


# Option Three
median = housing['RM'].median()
median


# In[33]:


#housing['RM'].fillna(median) # Housing data remain unchanged because we have assing


# In[34]:


housing.describe() # Before filling missing values


# In[35]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = 'median')
imputer.fit(housing)


# In[36]:


imputer.statistics_


# In[37]:


X = imputer.transform(housing)


# In[38]:


housing_tr = pd.DataFrame(X, columns= housing.columns)


# In[39]:


housing_tr.describe()


# ## Scikit-learn Design
# Primarily, Three types of object
# 1. Estimators : It estimats some parameter based on dataset. Exm imputer
# It has fit and transform method
# Fit method: fits the dataset and calculate internal parameters
# 2. Transformer: Transform methods take inputes and return output based on the learning from fit. It also has convenience function called fit_transform() which fit and transforms.
# 3. Predictors : Linearregressor is an example of predictors. Fit and predict are two common functions. It also give score function which evaluate the prediction.
# 

# ## Feature Scaling
# Primarily, there are two types of feature scaling method:
# 1. min-max scaling(Normalization)
# (values -min) / (max - min)  # Lies 0-1
# 2. Standardization:
# (values - mean / std)
# for this sklearn provide class standardScaler
# 

# ## Creating a Pipeline

# In[40]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy = 'median')),
    # Add as many as you can
    ('std_scaler', StandardScaler())
    
])


# In[41]:


housing_num_tr = my_pipeline.fit_transform(housing_tr)


# In[42]:


housing_tr.shape


# ## Selecting a Desired Model for Our Project

# In[43]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
#model = LinearRegression()
#model = DecisionTreeRegressor()
model = RandomForestRegressor()


# In[44]:


model.fit(housing_num_tr, housing_labels)


# ## After Trained Model

# In[45]:


some_testData = housing.iloc[:5]
some_test_labels = housing_labels.iloc[:5]


# In[46]:


prepared_data = my_pipeline.transform(some_testData)


# In[47]:


model.predict(prepared_data)


# In[48]:


#some_test_labels
list(some_test_labels)


# ## Evaluating Models

# In[49]:


from sklearn.metrics import mean_squared_error
housing_prediction  = model.predict(housing_num_tr)
lin_mse = mean_squared_error(housing_labels, housing_prediction)
lin_mse = np.sqrt(lin_mse)


# In[50]:


lin_mse # Liner Regresoor given: 23.33 not good, DTR,:0.0 (over fiiting)


# ## Using Better Eveluation Techniques:
# How it works:
# 1 2 3 4 5 : it create  5 group(cv : fold) (example it may more)
# it trains 2 3 4 5 and test 1
# agian it trains 1 3 4 5 and test 2
#  and so on , finalyy returns score

# In[51]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, housing_num_tr, housing_labels, scoring = "neg_mean_squared_error", cv = 10)


# In[52]:


rm_error = np.sqrt(-scores)# - because sqrt does not calculate negative value
rm_error


# In[53]:


def print_score(score):
    print("Score: ", score)
    print("Mean: ", score.mean())
    print("Std: ", score.std())


# In[54]:


print_score(rm_error)


# ## Saving Model

# In[60]:


from joblib import dump, load
dump(model,"housingModel.joblib")


# ## Tesing Model on test Data

# In[64]:


X_test = strat_test_set.drop('MEDV', axis = 1)
y_test = strat_test_set['MEDV'].copy()
X_test_prepared = my_pipeline.transform(X_test)
finale_prediction = model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, finale_prediction)


# In[67]:


final_rmse = np.sqrt(final_mse)
final_rmse


# In[69]:


finale_prediction


# In[72]:


prepared_data[0]

