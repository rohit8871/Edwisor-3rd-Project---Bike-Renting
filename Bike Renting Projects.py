#!/usr/bin/env python
# coding: utf-8

# # Project Name -  Bike Renting
# ## 1. Data Preprocessing

# In[3]:


# Importing the libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[4]:


# set working directory
os.chdir('C:/Users/Rohit/Desktop/Bike Renting Projects')


# In[5]:


# Importing the datasets
dataset= pd.read_csv('day.csv')
dataset.head()


# In[6]:


# Checking Null values 
dataset.isnull().sum().sum()


# In[7]:


#  Counting of Unique values of each column
dataset.nunique()


# ## Plotting Graphs

# In[8]:


## Importing Libraries for Visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# 
# ### Plot For "Season"

# In[9]:


# counting categories of season
plt.figure(figsize=(15,8))

sns.countplot(x='season',data=dataset)


# In[10]:


# Bar plot each categories of "season" with "count of total rental bikes (cnt)"
plt.figure(figsize=(15,8))

sns.barplot(x='season', y='cnt',data=dataset,ci=99)


# In[11]:


# plot for Column 'season'
plt.figure(figsize=(15,8))
# strip
sns.stripplot(x='season', y='cnt',data=dataset,jitter=0.4)


# In[29]:


# Box plot to visualize the concentration of data in each categories with "count of rantal Bikes"
plt.figure(figsize=(15,8))
sns.boxplot(x='season', y='cnt',data=dataset)


# ### Plot for "year"

# In[38]:


# counting categories of season
plt.figure(figsize=(15,8))
sns.countplot(x='yr',data=dataset)


# In[30]:


# Bar plot each categories of "year" with "count of total rental bikes (cnt)"
plt.figure(figsize=(15,8))

sns.barplot(x='yr', y='cnt',data=dataset,ci=99)


# In[34]:


# plot for Column 'year'
plt.figure(figsize=(15,8))
# strip
sns.stripplot(x='yr', y='cnt',data=dataset,jitter=0.4)


# In[37]:


# Box plot to visualize the concentration of data with "count of rantal Bikes"
plt.figure(figsize=(15,8))

sns.boxplot(x='yr', y='cnt',data=dataset)


# In[41]:


dataset.nunique()


# ### Plot for Month

# In[42]:


# counting categories of "month"
plt.figure(figsize=(15,8))

sns.countplot(x='mnth',data=dataset)


# In[43]:


# Bar plot each categories of "month" with "count of total rental bikes (cnt)"
plt.figure(figsize=(15,8))

sns.barplot(x='mnth', y='cnt',data=dataset,ci=99)


# In[45]:


#  Strip plot for Column 'month'
plt.figure(figsize=(15,8))
# strip
sns.stripplot(x='mnth', y='cnt',data=dataset,jitter=0.3)


# In[46]:


# Box plot to visualize the concentration of data in each categories with "count of rantal Bikes"
plt.figure(figsize=(15,8))
sns.boxplot(x='mnth', y='cnt',data=dataset)


# ### plot for holiday

# In[47]:


# counting categories of "holiday"
plt.figure(figsize=(15,8))

sns.countplot(x='holiday',data=dataset)


# In[48]:


# Bar plot each categories of "holiday" with "count of total rental bikes (cnt)"
plt.figure(figsize=(15,8))

sns.barplot(x='holiday', y='cnt',data=dataset,ci=99)


# In[49]:


# plot for Column 'holiday'
plt.figure(figsize=(15,8))
# strip
sns.stripplot(x='holiday', y='cnt',data=dataset,jitter=0.4)


# In[50]:


# Box plot to visualize the concentration of data in each categories with "count of rantal Bikes"
plt.figure(figsize=(15,8))

sns.boxplot(x='holiday', y='cnt',data=dataset)


# ### Plot for "weekday"

# In[52]:


# counting categories of "weekday"
plt.figure(figsize=(15,8))
sns.countplot(x='weekday',data=dataset)


# In[54]:


# Bar plot each categories of "weekday" with "count of total rental bikes (cnt)"
plt.figure(figsize=(15,8))

sns.barplot(x='weekday', y='cnt',data=dataset,ci=99)


# In[55]:


# plot for Column 'weekday '
plt.figure(figsize=(15,8))
# strip
sns.stripplot(x='weekday', y='cnt',data=dataset,jitter=0.4)


# In[56]:


# Box plot to visualize the concentration of data in each categories with "count of rantal Bikes"
plt.figure(figsize=(15,8))

sns.boxplot(x='weekday', y='cnt',data=dataset)


# In[57]:


dataset.nunique()


# ### Plot for "workingday"

# In[58]:


# counting categories of "workingday"
plt.figure(figsize=(15,8))

sns.countplot(x='workingday',data=dataset)


# In[59]:


# Bar plot each categories of "workingday" with "count of total rental bikes (cnt)"
plt.figure(figsize=(15,8))

sns.barplot(x='workingday', y='cnt',data=dataset,ci=99)


# In[60]:


# plot for Column 'workingday'
plt.figure(figsize=(15,8))
# strip
sns.stripplot(x='workingday', y='cnt',data=dataset,jitter=0.4)


# In[61]:


# Box plot to visualize the concentration of data in each categories with "count of rantal Bikes"
plt.figure(figsize=(15,8))

sns.boxplot(x='workingday', y='cnt',data=dataset)


# ### Plot for "weathersit"

# In[62]:


# counting categories of weathersit
plt.figure(figsize=(15,8))

sns.countplot(x='weathersit',data=dataset)


# In[63]:


# Bar plot each categories of "weathersit" with "count of total rental bikes (cnt)"
plt.figure(figsize=(15,8))

sns.barplot(x='weathersit', y='cnt',data=dataset,ci=99)


# In[13]:


# plot for Column 'weathersit'
plt.figure(figsize=(15,8))
# strip
sns.stripplot(x='weathersit', y='cnt',data=dataset,jitter=.4)


# In[65]:


# Box plot to visualize the concentration of data in each categories with "count of rantal Bikes"
plt.figure(figsize=(15,8))

sns.boxplot(x='weathersit', y='cnt',data=dataset)


# ### Plot for "temp"

# In[33]:


# Bar plot each categories of "temp" with "count of total rental bikes (cnt)"
plt.figure(figsize=(900,8))

sns.barplot(x='temp', y='cnt',data=dataset,ci=9)


# In[9]:


dataset['temp'].nunique()


# In[71]:


# plot for Column 'temp'
plt.figure(figsize=(15,8))
# strip
sns.stripplot(x='temp', y='cnt',data=dataset,jitter=0.0)


# ### Plot for "atemp"

# In[74]:


# plot for Column 'atemp'
plt.figure(figsize=(15,8))
# strip
sns.stripplot(x='atemp', y='cnt',data=dataset,jitter=0.4)


# In[11]:


dataset.head()


# ### Plot for "humidity"

# In[81]:


# plot for Column 'humidity'
plt.figure(figsize=(15,8))
# strip
sns.stripplot(x='hum', y='cnt',data=dataset,jitter=0.0)


# In[76]:


dataset.nunique()


# ### Plot for windspeed

# In[14]:


# plot for Column 'windspeed'
plt.figure(figsize=(15,8))
# strip
sns.stripplot(x='windspeed', y='cnt',data=dataset,jitter=0.1)


# ### Plot for "casual"

# In[89]:


# plot for Column 'casual'
plt.figure(figsize=(15,8))
# strip
sns.stripplot(x='casual', y='cnt',data=dataset,jitter=0.4)


# ###  Plot for "registered "

# In[24]:


# Bar plot each categories of "registered " with "count of total rental bikes (cnt)"
plt.figure(figsize=(300,8))

sns.barplot(x='registered', y='cnt',data=dataset,ci=99)


# In[96]:


# plot for Column 'registered '
plt.figure(figsize=(15,8))
# strip
sns.stripplot(x='registered', y='cnt',data=dataset,jitter=0.0)


# In[95]:


# Box plot to visualize the concentration of data in each categories with "count of rantal Bikes"
plt.figure(figsize=(15,8))

sns.boxplot(x='registered', y='cnt',data=dataset)


# #### check correlation between Independent Numeric variables

# In[9]:


# Dot Plot between 'registered ' and "casual"
plt.figure(figsize=(10,8))
# strip
sns.stripplot(x='registered', y='casual',data=dataset,jitter=0.0)


# ## 1.1) Missing Value Analysis

# In[12]:


# Checking Null values 
dataset.isnull().sum().sum()


# ## 1.2) Outlier Analysis

# 
# ## 1.2.a) Checking outlier values with Box Plot¶
# 

# In[13]:


# seperating categorical and numerical columns
numeric_columns=['temp', 'atemp', 'hum', 'windspeed','casual', 'registered']
categorical_columns=['instant','season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit']


# In[14]:


dataset.columns


# In[15]:


dataset.head()


# #### Now we will find out the outlier values one by one in "Numeric variables" with the help of BOX plot

# In[16]:


# Checking if there is any outlier values in dataset
for i in numeric_columns:

    sns.boxplot(dataset[i])
    plt.title("Checking outlier values for Variable " +str(i))
    plt.ylabel("Density")
    plt.show()


# In[176]:


# copy dataset
dataset_1 = dataset.copy()
#dataset = dataset_1.copy()


# In[18]:


# Below are the Numeric varible with outliers values
numeric_columns_outliers=['hum', 'windspeed','casual']


# ## 1.3.b) Removing outliers values

# In[19]:


# for i in numeric_columns_outliers:
#     print(i)
#     q75,q25=np.percentile(dataset.loc[:,i],[75,25])
#     
#     # Inter quartile range
#     iqr=q75-q25
#     
#     #Lower Fence
#     min=q25-(iqr*1.5)
#     # Upper fence
#     max=q75+(iqr*1.5)
#     print(min)
#     print(max)
#     
# # Droping outliers values
# dataset=dataset.drop(dataset[dataset.loc[:,i]<min].index)
# dataset=dataset.drop(dataset[dataset.loc[:,i]>max].index)


# In[20]:


dataset.shape


# ## 1.3.c) Detect and replace outliers with NA

# In[177]:


for i in numeric_columns_outliers:
    print(i)
    q75,q25=np.percentile(dataset.loc[:,i],[75,25])
    
    # Inter quartile range
    iqr=q75-q25
    
    #Lower Fence
    min=q25-(iqr*1.5)
    # Upper fence
    max=q75+(iqr*1.5)
    print(min)
    print(max)
    
    # Replacing all the outliers value to NA
    dataset.loc[dataset[i]< min,i] = np.nan
    dataset.loc[dataset[i]> max,i] = np.nan
# Checking if there is any missing value
dataset.isnull().sum().sum()


# In[178]:


missing_val=pd.DataFrame(dataset.isnull().sum())
missing_val


# In[179]:


#checking missing values after replacing outliers with NA through Heat Map
plt.figure(figsize=(10,5))
sns.heatmap(dataset.isnull(), cmap="viridis")
print("Total NA value in the dataset is =" +str(dataset.isnull().sum().sum()))


# In[180]:


# imputing numeric missing values by MEAN method 
for i in numeric_columns_outliers:
    print(i)
    dataset.loc[:,i]=dataset.loc[:,i].fillna(dataset.loc[:,i].mean())


# In[181]:


#checking missing values after Imputation by Mean Method
plt.figure(figsize=(10,5))
sns.heatmap(dataset.isnull(), cmap="viridis")
print("Total NA value in the dataset is =" +str(dataset.isnull().sum().sum()))


# In[182]:


# Checking if there is any outlier values in dataset
for i in numeric_columns_outliers:

    sns.boxplot(dataset[i])
    plt.title("Checking outlier values for Variable " +str(i))
    plt.ylabel("Density")
    plt.show()


# #####  we have removed almost outlier values

# ## 1.3 Feature Selection
# ### 1.3.1 Correlation Analysis

# In[183]:


from scipy import stats


# In[184]:


numeric_columns


# In[185]:


# Correlation Analysis
dataset_corr=dataset.loc[:,numeric_columns]

# set the width and height of plot
f, ax=plt.subplots(figsize=(10,10))

# Generate correlation Matrix
corr=dataset_corr.corr()

#plot using seaborn Library
sns.heatmap(corr,mask=np.zeros_like(corr,dtype=np.bool),            cmap=sns.diverging_palette(250,10,as_cmap=True),
           square=True,ax=ax,annot = True)


# ##### Here  "temp"  and  "atemp" are highly correlated  with correlation coeficient (r) = 0.99
# ##### Therefore we will remove any one out of them

# In[186]:


# Loading Library for calculating VIF of each numeric columns.
from statsmodels.stats.outliers_influence import variance_inflation_factor

# For each X, calculate VIF and save in dataframe
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(dataset[numeric_columns].values, i)                     for i in range(dataset[numeric_columns].shape[1])]
vif["features"] = dataset[numeric_columns].columns
vif


# In[187]:


###### Here, VIF for "temp" and "atemp" are high. This means both are highly correlated.


# ### 1.3.2 ANOVA Analysis

# In[188]:


categorical_columns1=['instant','season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit']

#loop for ANOVA test Since the target variable is continuous
for i in categorical_columns1:
    f, p = stats.f_oneway(dataset[i], dataset["cnt"])
    print("P value for variable "+str(i)+" is "+str(p))


# Here P value for all the variables are zero.
# 
# My Hypothesis is-
# 
# H0= Independent varaible does not explain our Target variable.
# 
# H1= Independent varaible explain our Target variable.
# 
# 
# If p< 0.05 then reject the NULL Hypothesis (H0), that means this particular Independent 
# variable is going to explain my Target Variable.
# 
# 
# Now, in my case -
# 
# P value for all the variables are zero.
# P<0.05 , that means all the categorical variables explain the target variable ("cnt")
# Therefore, I will not remove any categorical variable.
# 
# 
# conclusion-
# 
# we will Remove 'atemp'   on the basis of Correlation plot and vif values
# 
# &
# 
# we will also remove 'instant'  and "dteday"  whcih is irrelevant  for model learning & prediction

# In[189]:


# Droping the variables which are not Important
dataset_del = dataset.drop(['instant',"atemp", "dteday"], axis = 1)


# In[192]:


dataset_del.columns


# In[48]:


# copy dataset
dataset_2=dataset_del.copy()
#dataset_del=dataset_2.copy()


# ## 1.4 Feature scaling

# In[193]:


numeric_columns=['temp', 'hum', 'windspeed','casual', 'registered']


# In[194]:


dataset[numeric_columns].head()


# In[195]:


numeric_columns_scale=['casual', 'registered']


# In[196]:


# Checking if there is any normally distributed variable in data
for i in numeric_columns_scale:
    if i == 'cnt':
        continue
    sns.distplot(dataset[i],bins = 'auto')
    plt.title("Checking Distribution for Variable "+str(i))
    plt.ylabel("Density")
    plt.show()


# ###### columns "casual" and "resisters" are not scaled.

# In[197]:


# Since there is no normally distributed curve we will use Normalizationg for Feature Scalling
# #Normalization

for i in numeric_columns_scale:
    if i == 'Absenteeism time in hours':
        
        continue
    dataset_del[i] = (dataset_del[i] - dataset_del[i].min())/(dataset_del[i].max()-dataset_del[i].min())


# In[198]:


dataset_del[numeric_columns].head()


# In[199]:


# copying dataset
dataset_3=dataset_del.copy()
#dataset_del=dataset_3.copy()


# # 2 Machine Leaning Model

# In[200]:


# Important categorical variable to convert into dummy variables.
categorical_columns=['season','mnth', 'weekday', 'weathersit']
dataset_del[categorical_columns].nunique()


# In[201]:


# Get dummy variables for categorical variables
dataset_del = pd.get_dummies(data = dataset_del, columns = categorical_columns )

# Copying dataframe
dataset_4 = dataset_del.copy()


# In[202]:


dataset_del.head()


# In[203]:


dataset_del.nunique()


# In[204]:


### AVoiding dummy variable trap
# selecting columns to Avoid dummy variable trap
drop_columns=['season_1', 'mnth_1', 'weekday_0', 'weathersit_1']
dataset_del = dataset_del.drop(drop_columns, axis = 1)
dataset_del.shape


# In[205]:


dataset_del.head()


# In[73]:


# splitting the dataset into X and y
X = dataset_del.drop("cnt", axis = 1)
y = dataset_del.iloc[:,8].values


# In[74]:


X=pd.DataFrame(X)


# In[75]:


X.head()


# In[76]:


# splitting the dataset into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.20, random_state=0)


# In[77]:


X_train.shape


# In[78]:


X_test.shape


# ## 2.1) Multiple Linear Regression
# 

# In[79]:


# Importing Library for Linear Regression
from sklearn.linear_model import LinearRegression

# Fitting simple linear regression to the training data
regressor=LinearRegression()
LR_model=regressor.fit(X_train,y_train)


# In[80]:


# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = LR_model, X = X_train, y = y_train, cv =10 )
accuracy=accuracies.mean()
print(' Accuracy of Multiple Linear Regression ='+str(accuracy))
print(' Accuracy of all  the partitions='+str(accuracies))
print('Accuracy standard Deviation = ' + str(accuracies.std()))


# In[81]:


# Importing Library for Linear regression
import statsmodels.api as sm

# train the model using training set
LR_model = sm.OLS(y_train, X_train).fit()
LR_model.summary()


# In[82]:


# predicting the test set results
y_pred= LR_model.predict(X_test)
y_pred


# In[83]:


# Calculate MAPE
def mape(y_test,y_pred):
    mape = np.mean(np.abs((y_test - y_pred)/y_test))*100
    return mape

mape(y_test,y_pred)


# In[84]:


# Calculating RMSE for training data to check for over fitting
from sklearn.metrics import mean_squared_error
pred_train = LR_model.predict(X_train)
rmse_for_train = np.sqrt(mean_squared_error(y_train,pred_train))
print("Root Mean Squared Error For Training data = "+str(rmse_for_train))


# In[85]:


# Calculating RMSE for test data to check accuracy
y_pred= LR_model.predict(X_test)
rmse_for_test =np.sqrt(mean_squared_error(y_test,y_pred))
print("Root Mean Squared Error For Test data = "+str(rmse_for_test))


# In[86]:


# calculate R^2 value to check the goodness of fit
from sklearn.metrics import r2_score
print("R^2 Score(coefficient of determination) = "+str(r2_score(y_test,y_pred)))


# In[87]:


dataset.shape


# ##### There is little bit difference between RMSE train and RMSE test. so it might be overfitting case.
# #### We will try to reduce irrelevant features by using Principal component Analysis.

# ## 2.2  Decision Tree

# In[90]:


# Importing Library for Decision Tree
from sklearn.tree import DecisionTreeRegressor

# # Fitting Decision Tree Regression to the dataset
regressor = DecisionTreeRegressor(random_state = 0)
DT_model=regressor.fit(X_train, y_train)


# In[91]:


# Predicting a new result
y_pred = DT_model.predict(X_test)
y_pred


# In[92]:


y_test


# In[93]:


# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = DT_model, X = X_train, y = y_train, cv =10 )
accuracy=accuracies.mean()
print(' Accuracy ='+str(accuracy))
print(' Accuracy of all  the partitions='+str(accuracies))
print('Accuracy standard Deviation = ' + str(accuracies.std()))


# In[94]:


# Calculate MAPE
def mape(y_test,y_pred):
    mape = np.mean(np.abs((y_test - y_pred)/y_test))*100
    return mape

mape(y_test,y_pred)


# In[95]:


# Calculating RMSE for training data to check for over fitting
from sklearn.metrics import mean_squared_error
pred_train = DT_model.predict(X_train)
rmse_for_train = np.sqrt(mean_squared_error(y_train, pred_train))
print("Root Mean Squared Error For Training data = "+str(rmse_for_train))


# In[96]:


# Calculating RMSE for test data to check accuracy
y_pred= DT_model.predict(X_test)
rmse_for_test =np.sqrt(mean_squared_error(y_test,y_pred))
print("Root Mean Squared Error For Test data = "+str(rmse_for_test))


# In[97]:


# calculate R^2 value to check the goodness of fit
from sklearn.metrics import r2_score
print("R^2 Score(coefficient of determination) = "+str(r2_score(y_test,y_pred)))


# ##### Difference of RMSE train and test is High. Therefore, Decision Tree model is overfitting.

# ## 2.3 Random Forest

# In[106]:


# Importing Library for Random Forest
from sklearn.ensemble import RandomForestRegressor
# Fitting Random Forest Regression to the dataset
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
RF_model=regressor.fit(X_train, y_train)


# In[107]:


# Predicting a new result
y_pred = regressor.predict(X_test)
y_pred


# In[108]:


y_test


# In[109]:


# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = RF_model, X = X_train, y = y_train, cv =10 )
accuracy=accuracies.mean()
print(' Accuracy ='+str(accuracy))
print(' Accuracy of all  the partitions='+str(accuracies))
print('Accuracy standard Deviation = ' + str(accuracies.std()))


# In[110]:


# Calculate MAPE
def mape(y_test,y_pred):
    mape = np.mean(np.abs((y_test - y_pred)/y_test))*100
    return mape

mape(y_test,y_pred)


# In[111]:


# Calculating RMSE for training data to check for over fitting
from sklearn.metrics import mean_squared_error
pred_train = RF_model.predict(X_train)
rmse_for_train = np.sqrt(mean_squared_error(y_train,pred_train))
print("Root Mean Squared Error For Training data = "+str(rmse_for_train))


# In[112]:


# Calculating RMSE for test data to check accuracy
y_pred= RF_model.predict(X_test)
rmse_for_test =np.sqrt(mean_squared_error(y_test,y_pred))
print("Root Mean Squared Error For Test data = "+str(rmse_for_test))


# In[113]:


# calculate R^2 value to check the goodness of fit
from sklearn.metrics import r2_score
print("R^2 Score(coefficient of determination) = "+str(r2_score(y_test,y_pred)))


# ##### Difference of RMSE train and test is High. Therefore, Random Forest model is overfitting.

# ## 2.4 Gradiet Boosting

# In[114]:


# Importing library for Gradient Boosting
from sklearn.ensemble import GradientBoostingRegressor
# Building model on top of training dataset
GB_model = GradientBoostingRegressor().fit(X_train, y_train)


# In[115]:


# Predicting a new result
y_pred = GB_model.predict(X_test)
y_pred


# In[116]:


# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = GB_model, X = X_train, y = y_train, cv =10 )
accuracy=accuracies.mean()
print(' Accuracy ='+str(accuracy))
print(' Accuracy of all  the partitions='+str(accuracies))
print('Accuracy standard Deviation = ' + str(accuracies.std()))


# In[117]:


# Calculate MAPE
def mape(y_test,y_pred):
    mape = np.mean(np.abs((y_test - y_pred)/y_test))*100
    return mape

mape(y_test,y_pred)


# In[118]:


# Calculating RMSE for training data to check for over fitting
pred_train = GB_model.predict(X_train)
rmse_for_train = np.sqrt(mean_squared_error(y_train,pred_train))
print("Root Mean Squared Error For Training data = "+str(rmse_for_train))
 


# In[119]:


# Calculating RMSE for test data to check accuracy
pred_test = GB_model.predict(X_test)
rmse_for_test =np.sqrt(mean_squared_error(y_test,pred_test))
print("Root Mean Squared Error For Test data = "+str(rmse_for_test))


# In[120]:


# calculate R^2 value to check the goodness of fit
print("R^2 Score(coefficient of determination) = "+str(r2_score(y_test,pred_test)))


# ##### Difference of RMSE train and test is High. Therefore, Gradient Boosting model is overfitting.

# # 2.2 Support vector Regressor

# In[122]:


# Importing Library for SVR
from sklearn.svm import SVR

# Fitting SVR to the dataset
regressor = SVR(kernel = 'rbf')
SVR_model=regressor.fit(X_train,y_train)


# In[123]:


# predicting the test results
y_pred= SVR_model.predict(X_test)
y_pred


# In[124]:


# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = regressor, X = X_train, y = y_train, cv =10 )
accuracy=accuracies.mean()
print(' Accuracy ='+str(accuracy))
print(' Accuracy of all  the partitions='+str(accuracies))
print('Accuracy standard Deviation = ' + str(accuracies.std()))


# In[125]:


# Calculating RMSE for test data to check accuracy
from sklearn.metrics import mean_squared_error
y_pred= SVR_model.predict(X_test)
rmse_for_test =np.sqrt(mean_squared_error(y_test,y_pred))
print("Root Mean Squared Error For Test data = "+str(rmse_for_test))


# In[126]:


# Calculating RMSE for training data to check for over fitting
from sklearn.metrics import mean_squared_error
pred_train = SVR_model.predict(X_train)
rmse_for_train = np.sqrt(mean_squared_error(y_train,pred_train))
print("Root Mean Squared Error For Training data = "+str(rmse_for_train))


# In[127]:


# calculate R^2 value to check the goodness of fit
from sklearn.metrics import r2_score
print("R^2 Score(coefficient of determination) = "+str(r2_score(y_test,y_pred)))


# #### SVR is very poor learner. we will see it again after PCA

# # 3. Principal component analysis

# In[128]:


# Copying dataframe
#dataset1 = dataset.copy()
#dataset = dataset1.copy()
dataset_del.shape
X_test.shape


# In[129]:


# splitting the dataset into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.20, random_state=0)

#----------------------------------------------------------------------------------------------------#



# Performing PCA on Train and test data seperately
from sklearn.decomposition import PCA

#Data has 92 variables so no of components of PCA = 92
pca=PCA(n_components=30)

# Fitting this object to the Training and test set
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# The amount of variance that each PC explains
explained_variance = pca.explained_variance_ratio_

# Cumulative Variance explains
var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)

plt.plot(var1)
plt.show()

explained_variance


# In[130]:


# From the above plot selecting 26 components since it explains almost 90+ % data variance
pca = PCA(n_components=26)

# Fitting the selected components to the data
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)


# In[131]:


# Building the optimal model using Backward elimination
import statsmodels.formula.api as sm

X_train=np.append(arr=np.ones((584,1)).astype(int), values=X_train ,axis=1)
X_train=pd.DataFrame(X_train)
# selecting training columns
X_opt = X_train.iloc[:, 0:30]

regressor_OLS=sm.OLS(endog=y_train, exog=X_opt).fit()
regressor_OLS.summary()


# In[132]:


X_train.shape


# In[133]:


# Dropping irrelevant columns in Training dataset whose p value is highest
X_train = X_train.drop([16], axis = 1)
X_train.shape


# In[134]:


# Building the optimal model using Backward elimination
import statsmodels.formula.api as sm

# selecting training columns
X_opt = X_train.iloc[:, 0:30]

regressor_OLS=sm.OLS(endog=y_train, exog=X_opt).fit()
regressor_OLS.summary()


# In[135]:


# Building the optimal model using Backward elimination
import statsmodels.formula.api as sm

X_test=np.append(arr=np.ones((147,1)).astype(int), values=X_test ,axis=1)
X_test=pd.DataFrame(X_test)
# selecting training columns
X_opt = X_test.iloc[:, 0:30]

regressor_OLS=sm.OLS(endog=y_test, exog=X_opt).fit()
regressor_OLS.summary()


# In[136]:


# Dropping the same columns in  test dataset
X_test=pd.DataFrame(X_test)
X_test = X_test.drop([16], axis = 1)
X_test.shape


# In[137]:


X_train=pd.DataFrame(X_train)
X_train.shape


# In[138]:


y.shape


# In[139]:


y_test.shape


# # 3.a) Multiple Linear Regression after Dimensionality 
# #       Reduction

# In[140]:


# Importing Library for Linear Regression
from sklearn.linear_model import LinearRegression

# Fitting simple linear regression to the training data
regressor=LinearRegression()
LR_model=regressor.fit(X_train,y_train)


# In[141]:


# predicting the test set results
y_pred= LR_model.predict(X_test)
y_pred


# In[142]:


# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = LR_model, X = X_train, y = y_train, cv =10 )
accuracy=accuracies.mean()
print(' Accuracy of Multiple Linear Regression ='+str(accuracy))
print(' Accuracy of all  the partitions='+str(accuracies))
print('Accuracy standard Deviation = ' + str(accuracies.std()))


# In[143]:


# Importing Library for Linear regression
import statsmodels.api as sm

# train the model using training set
LR_model = sm.OLS(y_train, X_train).fit()
LR_model.summary()


# In[144]:


# Calculate MAPE
def mape(y_test,y_pred):
    mape = np.mean(np.abs((y_test - y_pred)/y_test))*100
    return mape

mape(y_test,y_pred)


# In[145]:


# Calculating RMSE for training data to check for over fitting
pred_train = LR_model.predict(X_train)
rmse_for_train = np.sqrt(mean_squared_error(y_train,pred_train))
print("Root Mean Squared Error For Training data = "+str(rmse_for_train))


# In[146]:


# Calculating RMSE for test data to check accuracy
from sklearn.metrics import mean_squared_error
y_pred= LR_model.predict(X_test)
rmse_for_test =np.sqrt(mean_squared_error(y_test,y_pred))
print("Root Mean Squared Error For Test data = "+str(rmse_for_test))


# In[147]:


# calculate R^2 value to check the goodness of fit
from sklearn.metrics import r2_score
print("R^2 Score(coefficient of determination) = "+str(r2_score(y_test,y_pred)))


# # 3.b) Decision Tree after Dimensionality Reduction

# In[148]:


# Importing Library for Decision Tree
from sklearn.tree import DecisionTreeRegressor

# # Fitting Decision Tree Regression to the dataset
regressor = DecisionTreeRegressor(random_state = 0)
DT_model=regressor.fit(X_train, y_train)
DT_model


# In[149]:


# Predicting a new result
y_pred = DT_model.predict(X_test)
y_pred


# In[150]:


# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = DT_model, X = X_train, y = y_train, cv =5 )
accuracy=accuracies.mean()
print(' Accuracy of Decision Tree='+str(accuracy))
print(' Accuracy of all  the partitions='+str(accuracies))
print('Accuracy standard Deviation = ' + str(accuracies.std()))


# In[151]:


# Calculate MAPE
def mape(y_test,y_pred):
    mape = np.mean(np.abs((y_test - y_pred)/y_test))*100
    return mape

mape(y_test,y_pred)


# In[152]:


# Calculating RMSE for training data to check for over fitting
from sklearn.metrics import mean_squared_error
pred_train = DT_model.predict(X_train)
rmse_for_train = np.sqrt(mean_squared_error(y_train,pred_train))
print("Root Mean Squared Error For Training data = "+str(rmse_for_train))


# In[153]:


# Calculating RMSE for test data to check accuracy
y_pred= DT_model.predict(X_test)
rmse_for_test =np.sqrt(mean_squared_error(y_test,y_pred))
print("Root Mean Squared Error For Test data = "+str(rmse_for_test))


# In[154]:


# calculate R^2 value to check the goodness of fit
from sklearn.metrics import r2_score
print("R^2 Score(coefficient of determination) = "+str(r2_score(y_test,y_pred)))


# ### Decision tree model is completely overfitting

# # 3.c) Random Forest after Dimensionality Reduction

# In[155]:


# Importing Library for Random Forest
from sklearn.ensemble import RandomForestRegressor
# Fitting Random Forest Regression to the dataset
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
RF_model=regressor.fit(X_train, y_train)


# In[156]:


# Predicting a new result
y_pred = regressor.predict(X_test)
y_pred


# In[157]:


# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = RF_model, X = X_train, y = y_train, cv =5 )
accuracy=accuracies.mean()
print(' Accuracy of Random forest ='+str(accuracy))
print(' Accuracy of all  the partitions='+str(accuracies))
print('Accuracy standard Deviation = ' + str(accuracies.std()))


# In[158]:


# Calculate MAPE
def mape(y_test,y_pred):
    mape = np.mean(np.abs((y_test - y_pred)/y_test))*100
    return mape

mape(y_test,y_pred)


# In[159]:


# Calculating RMSE for training data to check for over fitting
from sklearn.metrics import mean_squared_error
pred_train = RF_model.predict(X_train)
rmse_for_train = np.sqrt(mean_squared_error(y_train,pred_train))
print("Root Mean Squared Error For Training data = "+str(rmse_for_train))


# In[160]:


# Calculating RMSE for test data to check accuracy
y_pred= RF_model.predict(X_test)
rmse_for_test =np.sqrt(mean_squared_error(y_test,y_pred))
print("Root Mean Squared Error For Test data = "+str(rmse_for_test))


# In[161]:


# calculate R^2 value to check the goodness of fit
from sklearn.metrics import r2_score
print("R^2 Score(coefficient of determination) = "+str(r2_score(y_test,y_pred)))


# # 3.d) Gradiet Boosting after Dimensionality Reduction

# In[162]:


# Importing library for Gradient Boosting
from sklearn.ensemble import GradientBoostingRegressor
# Building model on top of training dataset
GB_model = GradientBoostingRegressor().fit(X_train, y_train)


# In[163]:


# Predicting a new result
y_pred = GB_model.predict(X_test)
y_pred


# In[164]:


# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = GB_model, X = X_train, y = y_train, cv =10 )
accuracy=accuracies.mean()
print(' Accuracy of Gradiet Boosting ='+str(accuracy))
print(' Accuracy of all  the partitions='+str(accuracies))
print('Accuracy standard Deviation = ' + str(accuracies.std()))


# In[165]:


# Calculate MAPE
def mape(y_test,y_pred):
    mape = np.mean(np.abs((y_test - y_pred)/y_test))*100
    return mape

mape(y_test,y_pred)


# In[166]:


# Calculating RMSE for training data to check for over fitting
pred_train = GB_model.predict(X_train)
rmse_for_train = np.sqrt(mean_squared_error(y_train,pred_train))
print("Root Mean Squared Error For Training data = "+str(rmse_for_train))
 


# In[167]:


# Calculating RMSE for test data to check accuracy
pred_test = GB_model.predict(X_test)
rmse_for_test =np.sqrt(mean_squared_error(y_test,pred_test))
print("Root Mean Squared Error For Test data = "+str(rmse_for_test))


# In[168]:


# calculate R^2 value to check the goodness of fit
print("R^2 Score(coefficient of determination) = "+str(r2_score(y_test,pred_test)))


# ## 3.e) Support Vector Regression after Dimensionality Reduction¶

# In[169]:


# Importing Library for SVR
from sklearn.svm import SVR

# Fitting SVR to the dataset
regressor = SVR(kernel = 'rbf')
SVR_model=regressor.fit(X_train,y_train)
SVR_model


# In[170]:


# predicting the test results
y_pred= SVR_model.predict(X_test)
y_pred


# In[171]:


# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator =SVR_model, X = X_train, y = y_train, cv =5 )
accuracy=accuracies.mean()
print(' Accuracy of SVR_Model ='+str(accuracy))
print(' Accuracy of all  the partitions='+str(accuracies))
print('Accuracy standard Deviation = ' + str(accuracies.std()))


# In[172]:


# Calculating RMSE for training data to check for over fitting
from sklearn.metrics import mean_squared_error
pred_train = SVR_model.predict(X_train)
rmse_for_train = np.sqrt(mean_squared_error(y_train,pred_train))
print("Root Mean Squared Error For Training data = "+str(rmse_for_train))


# In[173]:


# Calculating RMSE for test data to check accuracy
from sklearn.metrics import mean_squared_error
y_pred= SVR_model.predict(X_test)
rmse_for_test =np.sqrt(mean_squared_error(y_test,y_pred))
print("Root Mean Squared Error For Test data = "+str(rmse_for_test))


# In[174]:


# calculate R^2 value to check the goodness of fit
from sklearn.metrics import r2_score
print("R^2 Score(coefficient of determination) = "+str(r2_score(y_test,y_pred)))


# In[ ]:


#### SVR is not a good model for this problem


# In[ ]:





# ## 4 Model Selection

# ## 4.a) Comparing Accuracy 
# #### Accuracy of Multiple Linear Regression =0.9619477139588094
# ####    Accuracy of Decision Tree=0.7629252720101468
# ####       Accuracy of Random forest =0.8882143785575272
# ####       Accuracy of Gradiet Boosting =0.9161169893888012
# ####   Accuracy of SVR_Model =0.0004138691477876266
# ## Multiple Linear Regression has highest Accuracy

#    

#   

# ## 4.b) Comparing RMSE value for all the Model

# ### Multiple Linear Regression
# #RMSE For Training data = 348.6541043531216
# #RMSE For Test data = 387.8544256571429
# ##### difference = 39.2003

# ### Decision tree
# #Root Mean Squared Error For Training data = 0.0
# #Root Mean Squared Error For Test data = 872.348507402487
# ##### difference = 872.348507402487

# ### Random forest
# #Root Mean Squared Error For Training data = 232.90808643404776
# #Root Mean Squared Error For Test data = 629.230001844388
# ##### difference = 396.322
# 

# ### Gradient Boosting
# #Root Mean Squared Error For Training data = 218.7014471159227
# #Root Mean Squared Error For Test data = 517.4852151253407
# ##### difference = 295.4473
# 
# 

# ### SVR
# Root Mean Squared Error For Training data = 1896.7030417458407
# 
# Root Mean Squared Error For Test data = 2055.676382291139
# 
# #### difference = 158.9733
# ### Here, we can observe - Multiple Linear Regression has less difference between RMSE train and RMSE test.
# ### Therefore, there is less chance of having Overfitting.

#  

# ## 4.c) Comparing R^2 and Adjusted R^2 in Linear Regression by OLS ()
# ### For Training dataset
# R-squared: 0.966 ---> This determines how close the data are fitted to the regresion line.It indicates the percentage of variation explained by regression line out of total variation.
# 
# Adjusted R^2: 0.965 ---> It indicates that the percentage of variation explained by only those independents variables that actually affects the target variable.
# 
# 
# 
# ## 4.d) Conclusion-
# ###  With the respest of all the evaluation Metrics and analysis we have concluded that Multiple Linear Regreesion is working well.

# In[ ]:





# In[ ]:





# In[ ]:




