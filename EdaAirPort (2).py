#!/usr/bin/env python
# coding: utf-8

# In[241]:


#https://towardsdatascience.com/exploratory-data-analysis-eda-python-87178e35b14
#https://thispointer.com/pandas-skip-rows-while-reading-csv-file-to-a-dataframe-using-read_csv-in-python/
#import the useful libraries.
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
#https://www.kaggle.com/what0919/diabetes-prediction


# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# In[243]:


data = pd.read_csv('/content/drive/MyDrive/airlinesmall.csv')


# In[244]:


# Printing the data
data


# In[245]:


data.info()


# In[246]:


#import the useful libraries.
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Read the file in data without first two rows as it is of no use.
#data = pd.read_csv('/content/drive/MyDrive/diabetes.csv', skiprows=[0,2,5])
data1 = pd.read_csv('/content/drive/MyDrive/airlinesmall.csv', skiprows=[16,22,17])
#data = pd.read_csv("marketing_analysis.csv",skiprows = 2)
#print the head of the data frame.
data1.head()


# In[247]:


data1


# In[248]:


# Drop the customer id as it is of no use.
data1.drop('Dest', axis = 1, inplace = True)


# In[249]:


data1


# In[250]:


# Drop the customer id as it is of no use.
data1.drop('TailNum', axis = 1, inplace = True)


# In[251]:


data1


# In[252]:


#Origin
data1.drop('Origin', axis = 1, inplace = True)


# In[253]:


data1


# In[254]:


#Extract job  & Education in newly from "jobedu" column.
#data['job']= data["jobedu"].apply(lambda x: x.split(",")[0])


# In[255]:


# Checking the missing values
data1.isnull().sum()

#CRSElapsedTime


# In[256]:


# Dropping the records with CRSElapsedTime missing in data dataframe.
data1 = data1[~data1.CRSElapsedTime.isnull()].copy()

# Checking the missing values in the dataset.
data1.isnull().sum()


#CRSElapsedTime


# In[257]:


# Find the mode of month in data
airplan_driven_mode = data1.Diverted.mode()[0]

# Fill the missing values with mode value of month in data.
data1.Diverted.fillna(airplan_driven_mode, inplace = True)

# Let's see the null values in the month column.
data1.Diverted.isnull().sum()


# In[258]:


# Find the mode of month in data
airplan_driven_mode = data1.DepTime.mode()[0]

# Fill the missing values with mode value of month in data.
data1.DepTime.fillna(airplan_driven_mode, inplace = True)

# Let's see the null values in the month column.
data1.DepTime.isnull().sum()


# In[259]:


data1


# In[260]:


#drop the records with response missing in data.
data1 = data1[~data1.DepTime.isnull()].copy()
# Calculate the missing values in each column of data frame
data1.isnull().sum()


# In[261]:


data1


# In[262]:


#groupby the response to find the mean of the salary with response no & yes separately.
data1.groupby('Cancelled')['DepTime'].mean()


# In[263]:


#groupby the response to find the mean of the salary with response no & yes separately.
data.groupby('Cancelled')['DepTime'].mean()


# In[264]:


#groupby the response to find the median of the salary with response no & yes separately.
data1.groupby('Distance')['Cancelled'].median()


# In[265]:


#groupby the response to find the median of the salary with response no & yes separately.
data.groupby('Distance')['Cancelled'].median()


# In[265]:





# In[266]:


data1.info()


# In[267]:


data1.Cancelled                    


# In[268]:


data1.FlightNum  


# In[269]:


#plot the box plot of salary for yes & no responses.
sns.boxplot(data1.AirTime, data1. DepDelay)
plt.show()


# In[270]:


#	Pregnancies	Glucose	BloodPressure	BMI	DiabetesPedigreeFunction	Age	Outcome
result = pd.pivot_table(data=data1, index='FlightNum', columns='Cancelled',values='DepTime')
print(result)

#create heat map of education vs marital vs response_rate
sns.heatmap(result, annot=True, cmap = 'RdYlGn', center=0.117)
plt.show()


# In[271]:


data1


# In[271]:





# In[271]:





# In[272]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[273]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as matplot

import re
import sklearn

import warnings
warnings.filterwarnings("ignore")

get_ipython().run_line_magic('matplotlib', 'inline')


# In[307]:


data1.describe()


# In[309]:


data1.info()


# In[311]:


#data1.drop(['UniqueCarrier'], axis = 1, inplace=True)
data1.drop(['CancellationCode'], axis = 1, inplace=True)


# In[312]:


df=data1
df.info()


# In[313]:


df.describe()


# In[314]:


from sklearn.feature_selection import VarianceThreshold

df.dropna(axis=1, how='all')
df.dropna(axis=0, how='all')
"""
df = df.rename(columns = {'Year' : 'Year_tr',
                          'Month' : 'Month_tr',
                          'DayofMonth' : 'DayofMonth_tr', 
                          'DayOfWeek' : 'DayOfWeek_tr',
                          'DepTime' : 'DepTime_tr',
                          'CRSDepTime' : 'CRSDepTime_tr',
                          'ArrTime' : 'ArrTime_tr',
                          'CRSArrTime' : 'CRSArrTime_tr',
                          'FlightNum' : 'FlightNum_tr',
                          'ActualElapsedTime' : 'ActualElapsedTime_tr',
                          'CRSElapsedTime' : 'CRSElapsedTime_tr',
                          'AirTime' : 'AirTime_tr',
                          'ArrDelay' : 'ArrDelay_tr',
                          'DepDelay' : 'DepDelay_tr',
                          'Distance' : 'Distance_tr',
                          'TaxiIn' : 'TaxiIn_tr',
                          'Cancelled' : 'Cancelled_tr',
                          'Diverted' : 'Diverted_tr',
                          'CarrierDelay' : 'CarrierDelay_tr',
                          'WeatherDelay' : 'WeatherDelay_tr',
                          'NASDelay' : 'NASDelay_tr',
                          'SecurityDelay' : 'SecurityDelay_tr',
                          'CancellationCode' : 'CancellationCode_tr',
                          'LateAircraftDelay' : 'LateAircraftDelay_tr'                        
                          })
"""


# In[315]:


#df = df.loc[:, ['Year_tr', 'Month_tr', 'DayofMonth_tr', 'DayOfWeek_tr','DepTime_tr',    'CRSDepTime_tr', 'ArrTime_tr', 'CRSArrTime_tr', 'FlightNum_tr',       'ActualElapsedTime_tr', 'CRSElapsedTime_tr','AirTime_tr', 'ArrDelay_tr', 'DepDelay_tr',            'Distance_tr', 'TaxiIn_tr', 'TaxiOut_tr', 'Cancelled_tr', 'CancellationCode_tr',  'Diverted_tr',               'CarrierDelay_tr', 'WeatherDelay_tr', 'NASDelay_tr', 'SecurityDelay_tr', 'LateAircraftDelay_tr']]
#Cancelled
df = df.loc[:, ['DepTime', 'CRSDepTime']]
#	Year	Month	DayofMonth	DayOfWeek	DepTime	CRSDepTime	ArrTime	CRSArrTime	FlightNum	ActualElapsedTime	CRSElapsedTime	AirTime	ArrDelay	DepDelay	Distance	TaxiIn	TaxiOut	Cancelled	CancellationCode	Diverted	CarrierDelay	WeatherDelay	NASDelay	SecurityDelay	LateAircraftDelay
#	Year	Month	DayofMonth	DayOfWeek	DepTime	CRSDepTime	ArrTime	CRSArrTime	FlightNum	ActualElapsedTime	CRSElapsedTime	AirTime	ArrDelay	DepDelay	Distance	TaxiIn	TaxiOut	Cancelled	CancellationCode	Diverted	CarrierDelay	WeatherDelay	NASDelay	SecurityDelay	LateAircraftDelay
df.describe()


# In[316]:


from sklearn.feature_selection import VarianceThreshold

#year in us -> american : 0, not american : 1
df.dropna(axis=1, how='all')
df.dropna(axis=0, how='all')


# In[317]:


df['DepTime'] = df['DepTime'].fillna(value = 0)


# In[318]:


df.describe()


# In[320]:


data1.loc[data1['DepTime'] <= 6, 'DepTime'] = 0.0
data.head(10)


# In[305]:


data1.info()


# In[321]:


colormap = plt.cm.viridis
plt.figure(figsize=(10,10))
sns.heatmap(data1.astype(float).drop(axis=1, labels='DepTime').corr(), linewidths=0.1, vmax=1.0, square=True, cmap=colormap, annot=True)


# In[324]:


#show = sns.pairplot(df.drop(['ID', 'GlycoHemoglobin'], axis=1), hue='Diabetes', size=1.5, diag_kind='kde')
show = sns.pairplot(df, hue='DepTime', size=3.5, diag_kind='kde')

show.set(xticklabels=[])


# In[325]:


#show = sns.pairplot(df.drop(['ID', 'GlycoHemoglobin'], axis=1), hue='Diabetes', size=1.5, diag_kind='kde')
show = sns.pairplot(df, hue='CRSDepTime', size=3.5, diag_kind='kde')

show.set(xticklabels=[])


# In[331]:


from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
#from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[336]:


sns.lmplot(x ="DepTime", y ="Distance", data = data1, order = 2, ci = None)


# In[340]:


import pandas as pd

def clean_dataset(data1):
    assert isinstance(data1, pd.DataFrame), "data1 needs to be a pd.DataFrame"
    data1.dropna(inplace=True)
    indices_to_keep = ~data1.isin([np.nan, np.inf, -np.inf]).any(1)
    return data1[indices_to_keep].astype(np.float64)


# In[346]:


X = np.array(data1['DepTime']).reshape(-1, 1)
y = np.array(data1['Distance']).reshape(-1, 1)
  
# Separating the data into independent and dependent variables
# Converting each dataframe into a numpy array 
# since each dataframe contains only one column
data1.dropna(inplace = True)
  
# Dropping any rows with Nan values
train_X, test_X, train_Y, test_Y = train_test_split(X, y, test_size = 0.25)
  
# Splitting the data into training and testing data
regr = LinearRegression()
  
regr.fit(train_X, train_Y)
print(regr.score(test_X, test_Y))


# In[347]:


y_pred = regr.predict(test_X)
plt.scatter(test_X, test_Y, color ='b')
plt.plot(test_X, y_pred, color ='k')
  
plt.show()
# Data scatter of predicted values


# In[348]:


#create linear regression obj
lr_regr = linear_model.LinearRegression()

#training via linear regression model
lr_regr.fit(train_X, train_Y)

#make prediction using the test set
lr_pred_diabetes = lr_regr.predict(test_X)
lr_score = lr_regr.score(test_X, test_Y)

print('LRr_Coefficients: ', lr_regr.coef_)
print('LR_Mean Square Error: %.2f' % mean_squared_error(test_Y, lr_pred_diabetes))
print('LR_Variance score: %.2f' % r2_score(test_Y, lr_pred_diabetes))
print('Score: %.2f' % lr_regr.score(test_X, test_Y))


# In[ ]:




