#!/usr/bin/env python
# coding: utf-8

# # Used Car Value
# 
# Using Machine Learning to predict the price of a used car based on past data.
# 
# * Used car prices depend on the kilometers driven, the age of the car, brand and model along with several other factors.
# * Based on past data, a regression model can predict / estimate the value of a used car.
# 
# The following code / steps are for developing a Machine Learning model to predict the value of used cars.
# 
# The original dataset has been taken from https://www.kaggle.com/orgesleka/used-cars-database
# 
# This model can be used for future used car value predictions also, using a similarly formatted dataset.

# In[2]:


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib as jl
from sklearn.linear_model import LinearRegression
# explicitly require this experimental feature for using HistGradientBoostingRegressor
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import precision_recall_fscore_support


# The import statements above are collated as per the requirements of the methods used throughout the notebook

# Uncomment the following 2 lines if you choose to execute this notebook in Google Colab

# In[3]:


#from google.colab import files
#uploaded = files.upload()


# In[4]:


import pandas as pd
auto_df = pd.read_csv("autos.csv")


# Looking at the data, most of the features are categorical.
# We will have to convert these to numerical values.
# We also will have to remove the features which do not have any impact on the price.

# In[5]:


auto_df


# In[6]:


print(auto_df.shape)


# In[7]:


auto_df.describe()


# As can be seen above, some values are erroneous.
# 
# For example:
# 
# min yearOfRegistration: 1000 or max yearOfRegistration: 9999
# 
# min price: 0 or max price: 2.14x10^9
# 
# min powerPS: 0 or max powerPS: 20000

# In[8]:


auto_df.columns


# In[9]:


print("Number of records: ", len(auto_df))


# ### Cleaning data
# *  Removing unwanted records / outliers
# *  Removing records of vehicles that are too old (older than 1999).
# *  Removing records having invalid (future) date.
# *  Removing records with too low a price.

# In[10]:


# Ignoring cars older than year 2000 and future dates/invalid dates and 
# those with too low a price
auto_df = auto_df.where((auto_df['yearOfRegistration'] > 1999) & 
                        (auto_df['yearOfRegistration'] < 2020) & 
                        (auto_df['price'] > 20)).dropna()


# In[11]:


print(auto_df['yearOfRegistration'].describe())


# #### Removing records having likely incorrect price
# 

# In[12]:


# Removing records having likely incorrect price
indexNames = auto_df[ auto_df['price'] == 11111111.0	 ].index
auto_df.drop(indexNames , inplace=True)
indexNames = auto_df[ auto_df['price'] == 12345678.0	 ].index
auto_df.drop(indexNames , inplace=True)
indexNames = auto_df[ auto_df['price'] == 999999.0	 ].index
auto_df.drop(indexNames , inplace=True)
indexNames = auto_df[ auto_df['price'] == 9999999.0	 ].index
auto_df.drop(indexNames , inplace=True)
indexNames = auto_df[ auto_df['price'] >= 2000000.0	 ].index
auto_df.drop(indexNames , inplace=True)


# In[13]:


print(auto_df['price'].describe())


# In[14]:


print("Number of Records: ", len(auto_df))


# In[15]:


print(auto_df.dtypes)


# In[16]:


# Converting float types to int
auto_df["price"] = auto_df["price"].astype(int)
auto_df["powerPS"] = auto_df["powerPS"].astype(int)
auto_df["kilometer"] = auto_df["kilometer"].astype(int)
auto_df["yearOfRegistration"] = auto_df["yearOfRegistration"].astype(int)
auto_df["monthOfRegistration"] = auto_df["monthOfRegistration"].astype(int)


# In[17]:


print(auto_df.dtypes)


# ### Feature Engineering
# * Converting **yeafOfRegistration** and **monthOfRegistration** to a new feature **age**
# * Creating new feature **name_len** from the existing feature **name**

# In[18]:


# function that returns calculates and returns the age of the vehicle.
def age_calculator(df):
    # Setting monthOfRegistration having value 0 to January (1)
    df["monthOfRegistration"].replace({0: 1}, inplace=True)
    
    # Creating feature dateOfRegistration from yearOfRegistration and monthOfRegistration
    df["dateOfRegistration"] = df["yearOfRegistration"].astype(str) + df["monthOfRegistration"].astype(str) + "1"
    
    # Converting dateOfRegistration to date format
    df['dateOfRegistration'] = pd.to_datetime(df['dateOfRegistration'], 
                                               format='%Y%m%d')
    # Calculating age and creating age feature from dateOfRegistration
    from datetime import datetime
    from datetime import date
    def calculate_age(dt):
        today = date.today()
        return today.year - dt.year - ((today.month, today.day) < (dt.month, dt.day))
    
    # Creating a new feature "age" based on dateOfRegistration
    return df['dateOfRegistration'].apply(calculate_age)

auto_df['age'] = age_calculator(auto_df)


# In[19]:


# Count of cars based on the year of registration (to further narrow the data set)
auto_df['age'].value_counts()[:28].plot(kind='bar')


# In[20]:


auto_df['age'].describe()


# In[21]:


auto_df.isnull().sum()


# In[22]:


print(len(auto_df["name"].unique()))


# The name feature has too many unique records and thus does not have any bearing on the price.
# But the length of the name could have some correlation with price given that longer name length indicates more features.

# In[23]:


auto_df['name_len'] = auto_df['name'].str.len()
print(auto_df['name_len'].unique())


# In[24]:


# removing outlier with too long name length.
indexNames = auto_df[auto_df['name_len'] >= 200].index
auto_df.drop(indexNames , inplace=True)
print(auto_df['name_len'].unique())


# In[25]:


auto_df.plot.scatter(x='name_len', y='price', c='DarkBlue')


# In[26]:


# plotting the price of the vehicle against the powerPS
auto_df.plot.scatter(x='powerPS', y='price', c='DarkBlue')


# We see there are many invalid entries for powerPS (either too high or too low)

# In[27]:


print("Number of rows having power > 600: ", len(auto_df.where(auto_df.powerPS > 600).dropna()))
print("Number of rows having power < 30: ", len(auto_df.where(auto_df.powerPS < 30).dropna()))


# In[28]:


# Removing records with invalid powerPS values (>= 600 or < 30)
indexNames = auto_df[ auto_df['powerPS'] >= 600 ].index
auto_df.drop(indexNames , inplace=True)
indexNames = auto_df[ auto_df['powerPS'] < 30 ].index
auto_df.drop(indexNames , inplace=True)


# In[29]:


auto_df.plot.scatter(x='powerPS', y='price', c='DarkBlue')


# In[30]:


# plotting price of used car against the age
auto_df.plot.scatter(x='age', y='price', c='DarkBlue')


# In[31]:


# plotting price of used car against the brand
auto_df.plot.scatter(x='price', y='brand', c='DarkBlue', figsize=(7, 12))


# In[32]:


# plotting the price of used car against the model
auto_df.plot.scatter(x='price', y='model', c='DarkBlue', figsize=(7, 30))


# In[33]:


# plotting price of used car against the vehicleType
auto_df.plot.scatter(x='vehicleType', y='price', c='DarkBlue')


# In[34]:


# plotting price of used car against the kilometer
auto_df.plot.scatter(x='kilometer', y='price', c='DarkBlue')


# In[35]:


# plotting price of used car against the notRepairedDamage
auto_df.plot.scatter(x='notRepairedDamage', y='price', c='DarkBlue')


# As can be seen from above plots, the following features do have some impact on the price of the car.
# 
# notRepairedDamage (nien or no means the price is likely to be higher)
# 
# Lesser kilometer driven fetches more price
# 
# vehicleType also has an affect on the price - coupes are generally costlier
# 
# model also has some impact on price - 911's are generally costlier
# 
# brand also has slight correlation with price - Mercedes, BMWs and Porsche are usually costlier than Nissan
# 
# age also has a bearing on the price. Older the car cheaper it is.
# 
# powerPS has strong correlation to price
# 
# name_len also has some bearing on the price as is evident from the plot.
# 

# In[36]:


# Following columns have no bearing on the price.
print(auto_df.offerType.unique())
print(auto_df.seller.unique())
print(auto_df.abtest.unique())
print(auto_df.nrOfPictures.unique())


# In[37]:


print(auto_df.offerType.value_counts())


# In[38]:


print(auto_df.seller.value_counts())


# In[39]:


print(auto_df.abtest.value_counts())


# In[40]:


print(auto_df.nrOfPictures.value_counts())


# In[41]:


auto_df.plot.scatter(x='abtest', y='price', c='DarkBlue')


# offerType has only two values and there is only 1 record having value Gesuch.
# 
# Hence this feature can be dropped
# 
# seller also has only 2 values and only 1 record having value gewerblich.
# 
# Hence I will drop this feature also
# 
# nrOfPictures has only 1 value (0.0) and is thus of no use.
# 
# I will drop this too. 
# 
# Abtest has two values  (test and control) and they are almost similar in count.
# 
# Moreover, from the plot, we can see the price range is equally distributed across two values of abtest (except for an outlier).
# 
# Hence this might be of importance as well, so I will drop this feature as well.
# 

# ### Encoding
# 
# Mapping and replacing the categorical values with numbers

# In[42]:


# Replacing fuel types with numerals
auto_df['fuelType'].unique()
auto_df_fuelType_dic = dict(zip(list(auto_df['fuelType'].unique()), list(range(1, len(auto_df['fuelType'].unique())+1))))
auto_df["fuelType"].replace(auto_df_fuelType_dic, inplace=True)


# In[43]:


# Replacing gearbox and notRepairedDamage with numerals
auto_df["gearbox"].replace({"manuell": 1, "automatik": 2}, inplace=True)
auto_df["notRepairedDamage"].replace({"ja": 1, "nein": 2}, inplace=True)


# In[44]:


# Replacing brand by numbers
auto_df_brand_dic = dict(zip(list(auto_df['brand'].unique()), list(range(1, len(auto_df['brand'].unique())+1))))
auto_df["brand"].replace(auto_df_brand_dic, inplace=True)

# Replacing model with numbers
auto_df_models_dic = dict(zip(list(auto_df['model'].unique()), list(range(1, len(auto_df['model'].unique())+1))))
auto_df["model"].replace(auto_df_models_dic, inplace=True)

# Replacing vehicleType by numbers
auto_vehicleType_dic = dict(zip(list(auto_df['vehicleType'].unique()), list(range(1, len(auto_df['vehicleType'].unique())+1))))
auto_df["vehicleType"].replace(auto_vehicleType_dic, inplace=True)


# In[79]:


# Mapping of categorical values and corresponding numbers
print("fuelTypes Mapping: ", auto_df_fuelType_dic)
print("\ngearbox mapping: {manuell: 1, automatik: 2}")
print("\nnotRepairedDamage: {ja: 1, nein: 2}")
print("\nbrand", auto_df_brand_dic)
print("\nmodel", auto_df_models_dic)
print("\nvehicleType", auto_vehicleType_dic)


# ### Feature selection
# Dropping features which do not have any impact on the price

# In[45]:


print(auto_df.columns)


# In[46]:


# Dropping features which don't have any correlation to the price
# or are redundant due to feature engineering done earlier
auto_df.drop(['dateCrawled',
              'name',
              'seller', 
              'offerType', 
              'abtest',
              'nrOfPictures',
              'lastSeen',
              'dateCreated',
              'postalCode',
              'yearOfRegistration',
              'monthOfRegistration',
              'dateOfRegistration'], axis='columns', inplace=True)


# In[47]:


print(auto_df.columns)


# In[48]:


print("Number of records: ", len(auto_df))


# In[49]:


# Checking for NaN values in the data
print(auto_df.isna().sum())


# In[50]:


print(auto_df.dtypes)


# #### Plotting the correlations

# In[51]:


# Visualizing the correlation heatmap
corr = auto_df.corr()
plt.figure(figsize = (10,10))
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 420, n=200),
    square=True,
    linewidths=1,
    annot=True,
    linecolor='White',
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);


# As we can see above, there is no strong correlation between the features
# 
# Given below is the correlation histogram of each feature with price.

# In[52]:


print("Correlation of features with price:\n", 
      auto_df.corr().loc[:,'price'].abs().sort_values(ascending=False)[1:])
auto_df.corr().loc[:,'price'].abs().sort_values(ascending=False)[1:].plot.bar()


# ### Modelling (Using Linear regression to set the baseline)
# 
# To guage the baseline, I will try Linear Regression with reduced feature set.
# 
# I will include **powerPS**, **age**, **kilometer**, **gearbox**, **vehicleType** and **name_len**

# In[53]:


# Assigning input features in variable X and Target (price) in Y
X = auto_df.drop(['price', 'brand', 'model', 'fuelType', 'notRepairedDamage', ],
                 axis='columns', inplace = False)
Y = auto_df["price"]


# ### Feature Scaling
# Given the large variation in Price, I am using logarithm (log1p function) to normalise the Price values

# In[54]:


# Normalizing the price range by taking a log of price values
Y = np.log1p(Y)

plt.figure(figsize=(10, 4))
prices = pd.DataFrame({"1. Before":auto_df['price'], "2. After":Y})
prices.hist()


# In[55]:


Y.describe()


# ### Training the base model (using Linear Regression)
# 
# Using this step, I will assess whether to include all the features we have shortlisted or only a reduced feature set is sufficient to build a viable model

# In[56]:


# Splitting the dataset into Training dataset and Test Dataset
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3)

# Trying Linear Regression model 
lr_model = LinearRegression()
lr_model.fit(X_train, Y_train)
print("R2_Score for Training data: ", lr_model.score(X_train, Y_train))
print("R2_Score for Test data: ", lr_model.score(X_test, Y_test))


# Using Linear Regression again by including additional features that we dropped before. i.e. **brand**, **model**, **fuelType**, **notRepairedDamage**

# In[57]:


# Assigning input features in variable X and Target (price) in Y
X = auto_df.drop(['price'], axis='columns', inplace = False)
Y = auto_df['price']
# Normalizing the price range by taking a log of price values
Y = np.log1p(Y)


# In[58]:


# Splitting the dataset into Training dataset and Test Dataset
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3)

lr_model = LinearRegression()
lr_model.fit(X_train, Y_train)
print("R2_Score for Training data: ", lr_model.score(X_train, Y_train))
print("R2_Score for Test data: ", lr_model.score(X_test, Y_test))


# ### Model Selection
# We see above that including more features improved the score.
# 
# Hence we will proceed with the below feature set as final and use **KFold Cross Validation** technique to choose a better regression algorithm.
# 
# *   powerPS
# *   age
# *   kilometer
# *   gearbox
# *   vehicleType
# *   name_len
# *   notRepairedDamage
# *   fuelType
# *   model
# *   brand

# In[59]:


models = []
models.append(('LR', LinearRegression()))
models.append(('HGBR', HistGradientBoostingRegressor()))
models.append(('GBR', GradientBoostingRegressor()))
models.append(('RFR', RandomForestRegressor()))
models.append(('XGBR', XGBRegressor(objective='reg:squarederror')))
# for XGBRegressor, we have to use the package xgboost and that must be 
# installed when we install sklearn and other packages


# In[60]:


print(models)


# In[61]:


# Using K-Fold Cross-Validation to determine the performance of different algorithms
results = []
for name, model in models:
    cv_results = cross_val_score(model, X_train, Y_train, cv=KFold(n_splits=10))
    results.append(cv_results)
    print("%s: Mean %f, Standard Deviation (%f)" % (name, cv_results.mean(), cv_results.std()))


# We will try Linear Regression first to get the base model and plot some visualizations.

# In[62]:


lr_model = LinearRegression()
lr_model.fit(X_train, Y_train)
print("R2_Score for Training data: ", lr_model.score(X_train, Y_train))
print("R2_Score for Test data: ", lr_model.score(X_test, Y_test))


# In[63]:


# plotting the predicted values vs actual values with respect to 
# different features in test dataset

for i in X_train.columns:
    plt.scatter(X_test[i], Y_test, color = "red")
    plt.scatter(X_test[i], lr_model.predict(X_test), color = "green")
    plt.title("Price of Used Car with respect to " + i)
    plt.xlabel(i)
    plt.ylabel('Price')
    plt.show()


# As can be gleaned from these plots, Linear regression does an OK job of predicting the Used car price, but as indicated by KFold Cross Validation, we see other algorithms produce better score.

# ### GridSearchCV for selecting best Hyper Parameters
# 
# So I will use **HistGradientBoostingRegressor** because of better cv score and try to arrive at better parameters using GridSearchCV

# In[64]:


model_hist_gbr = HistGradientBoostingRegressor()
param_grid = {
    'max_iter': [1500, 2500, 3500],
    'min_samples_leaf': [20, 40],
    'learning_rate': [0.05, 0.1, 1]
}
gs_cv = GridSearchCV(model_hist_gbr, param_grid, n_jobs=6)


# In[65]:


gs_cv.fit(X_train, Y_train)
print(gs_cv.best_params_)


# In[66]:


print(gs_cv.best_estimator_)


# In[67]:


mae_HGB_train = mean_absolute_error(Y_train, gs_cv.predict(X_train))
print("HistGradientBoosting Training set Mean Absolute Error: %.4f" % mae_HGB_train)
rmse_HGB_train = mean_squared_error(Y_train, gs_cv.predict(X_train))
print("HistGradientBoosting Training set Mean Squared Error: %.4f" % rmse_HGB_train)
mae_HGB_test = mean_absolute_error(Y_test, gs_cv.predict(X_test))
print("HistGradientBoosting Test set Mean Absolute Error: %.4f" % mae_HGB_test)
rmse_HGB_test = mean_squared_error(Y_test, gs_cv.predict(X_test))
print("HistGradientBoosting Test set Mean squared Error: %.4f" % rmse_HGB_test)
print("HistGradientBoosting Training Set Score: ", gs_cv.score(X_train, Y_train))
print("HistGradientBoosting Test Set Score: ", gs_cv.score(X_test, Y_test))


# ## Results / Analysis
# 
# ### Predicted Price vs Actual Price

# In[68]:


print(len(X_test))


# The test data has 56012 rows.
# 
# Hence I will try to plot random 200 records, first 200 records as well as last 200 records

# In[69]:


# Comparison chart of predicted price (orange) vs actual listed price (blue)
Original_price = Y_test.reset_index(drop=True)
Predicted_price = pd.Series(gs_cv.predict(X_test)).reset_index(drop=True)
compare_df = pd.concat(
    [Original_price, Predicted_price], axis=1).rename(
    columns={'price': 'actual price', 0:'predicted price'})

# plotting random 200 entries
ax = compare_df.sample(n=200).reset_index(drop=True).plot(title="Random 200 entries", figsize=(20,5))
ax.set_xlabel("vehicle instance")
ax.set_ylabel("price - log1p value")

# plotting first 200 entries
ax = compare_df.head(n=200).plot(title="First 200 entries", figsize=(20,5))
ax.set_xlabel("vehicle instance")
ax.set_ylabel("price - log1p value")

# plotting last 200 entries
ax = compare_df.tail(n=200).plot(title="Last 200 entries", figsize=(20,5))
ax.set_xlabel("vehicle instance")
ax.set_ylabel("price - log1p value")


# In[70]:


# plotting random 50 entries
ax = compare_df.sample(n=50).reset_index(drop=True).plot.bar(title="Random 50 entries", figsize=(20,5))
ax.set_xlabel("vehicle instance")
ax.set_ylabel("price - log1p value")

# plotting first 50 entries
ax = compare_df.head(n=50).plot.bar(title="First 50 entries", figsize=(20,5))
ax.set_xlabel("vehicle instance")
ax.set_ylabel("price - log1p value")

# plotting last 50 entries
ax = compare_df.tail(n=50).plot.bar(title="Last 50 entries", figsize=(20,5))
ax.set_xlabel("vehicle instance")
ax.set_ylabel("price - log1p value")


# As indicated by above plots, this model produced by HistGradientBoostingRegressor provides a much better fit, and the predicted Used car prices closely match the actual listed prices in the test data.

# In[71]:


# plotting the predicted values vs actual values with respect to 
# different features in test dataset
for i in X_train.columns:
    plt.figure(figsize= (5, 4))
    plt.scatter(X_test[i], Y_test, color = "red")
    plt.scatter(X_test[i], gs_cv.predict(X_test), color = "green")
    plt.title("Price of Used Car with respect to " + i)
    plt.xlabel(i)
    plt.ylabel("Price")
    plt.show()


# In[72]:


# dump the pickle file so as to load the model later on.
jl.dump(gs_cv.best_estimator_, 'model_hist_gbr.pkl')


# ### Transformation / Predictor Function
# 
# In order to predict from actual data, we have to create a transformation function that can be used to transform raw data so that it can be consumed by the model.
# 
# The transformation function can be built using the variables already created in this notebook.
# 
# The steps include:
# 
# 
# * Remove features not included while training the model
# * Convert categorical values into numerical values as was done during data preparation
# 
# 
# 
# 

# In[73]:


def transformation(data):
    data["powerPS"] = data["powerPS"].astype(int)
    data["kilometer"] = data["kilometer"].astype(int)
    data["yearOfRegistration"] = data["yearOfRegistration"].astype(int)
    data["monthOfRegistration"] = data["monthOfRegistration"].astype(int)
    data['age'] = age_calculator(data)
    data['name_len'] = data['name'].str.len()
    data.drop(['dateCrawled',
               'name',
               'seller',
               'offerType',
               'abtest',
               'nrOfPictures',
               'lastSeen',
               'dateCreated',
               'postalCode',
               'yearOfRegistration',
               'monthOfRegistration',
               'dateOfRegistration'], axis='columns', inplace=True)
    data["fuelType"].replace(auto_df_fuelType_dic, inplace=True)
    data["gearbox"].replace({"manuell": 1, "automatik": 2}, inplace=True)
    data["notRepairedDamage"].replace({"ja": 1, "nein": 2}, inplace=True)
    data["brand"].replace(auto_df_brand_dic, inplace=True)
    data["model"].replace(auto_df_models_dic, inplace=True)
    data["vehicleType"].replace(auto_vehicleType_dic, inplace=True)


# In[74]:


def predictor(data):
    transformation(data)
    model = jl.load('model_hist_gbr.pkl')
    price = model.predict(data)
    price = np.expm1(price)
    return price


# In[75]:


auto_df_test = pd.read_csv("autos.csv")
# The max yearOfRegistration should be current year
auto_df_test = auto_df_test.where((auto_df_test['yearOfRegistration'] > 1999) & 
                        (auto_df_test['yearOfRegistration'] < 2020) & 
                        (auto_df_test['price'] > 20)).dropna().sample(n=20)
predicted_price = predictor(auto_df_test.drop(['price'], axis='columns'))
print(predicted_price)


# ## Conclusion
# 
# * For this dataset, the features affecting the price are:
# 	**powerPS, age, kilometer, gearbox, vehicleType, name_len, notRepairedDamage, fuelType, brand** and **model**.
# * Linear Regression allowed me to identify the features that have some impact on the price of the used car.
# 	It served as a base model against which I compared the other algorithms.
# * Using KFold Cross-Validation, **HistGradientBoostingRegressor** seems better performing algorithm for this problem.
# * Further tuning of HistGradientBoostingRegressor using GridSearchCV led to a model with improved R2_Score on test data and better price predictions of Used Cars.

# ## References
# 
# * HistGradientBoostingRegressor
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html
# 
# * Used Car Database:https://www.kaggle.com/orgesleka/used-cars-database
# 
# * K-Fold Cross-Validation:
# https://machinelearningmastery.com/k-fold-cross-validation/
# 
# * GridSearchCV:
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
# 

# In[ ]:




