import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

'''Problem Statement
Given a dataset which captures the salary from July 1st, 2013 through June 30th, 2014.
It includes only those employees who are employed on June 30, 2014. Predict the salary of Employees working in Baltimore.'''


df = pd.read_csv('salariesbaltimore.csv')
print(df)

# Making a copy of the original DataFrame
newdf = df.copy()

print(df.isnull().sum())

# Removing the leading & trailing spaces and converting all the columns into the lowercase. some col had space,ex- ' Name'
newdf.columns= newdf.columns.str.strip().str.lower()
print(newdf.columns)

# Delete the column GrossPay
newdf.drop('grosspay', axis=1, inplace=True)

# Values_counts for agencyid
print(newdf.agencyid.value_counts())#Return a Series containing counts of unique values.
print(newdf.agency.value_counts())#Return a Series containing counts of unique values.

print(newdf[newdf['agencyid']=='P04001']['agency'])

print(newdf.set_index('agencyid')['agency'])
#now sort above on the basis of agencyid that is repeated more no. of times
#print(newdf.set_index('agencyid')['agency'].value_counts().sort_index())
#print(newdf[newdf['agencyid']=='P04001']['agency'])



# Remove the $ from the annualsalary column and change the dtype to integer
newdf['annualsalary'] = newdf['annualsalary'].str.strip('$').astype(float)
print(newdf)

x=newdf.iloc[:,-1].values.reshape(-1,1)
y=newdf.iloc[:,-1].values.reshape(-1,1)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

#model=LinearRegression()   #accuracy=1.0

model=SVR(kernel='rbf')

model.fit(x_train,y_train)

y_pred=model.predict(x_test)

print(y_test)
print()
print(y_pred)

print(r2_score(y_test,y_pred))


# Plot top 10 Jobs based on hiring
top10=newdf.groupby(['jobtitle'])['jobtitle'].value_counts().sort_values(ascending=False).head(10).plot.bar()
plt.show()
#print(top10)