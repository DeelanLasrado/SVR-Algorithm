import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('https://raw.githubusercontent.com/mk-gurucharan/Regression/master/SampleData.csv')

df.rename(columns={'Hours of Study':'Hours'}, inplace=True)
print(df.head())
print(df.isnull().sum())

plt.scatter(df.Hours, df.Marks)
plt.xlabel('Hours of Study')
plt.ylabel('Marks')
plt.title('Hours of Study V/s Marks')
plt.show()



#this is also right
'''x=np.array(df.Hours)
y=np.array(df.Marks)'''

print(df.iloc[:,:-1])  #it will provide col name along with the values
#X = df.iloc[row,col]
X = df.iloc[:,:-1].values #select all rows and All the col except last col        .values() is used only to copy the values and not the lebels
y = df.iloc[:,-1].values #only last col




#feature scalling- it will scale down all the values b/w -1 and 1, it will also increase the accuracy

from sklearn.preprocessing import StandardScaler

stanscale = StandardScaler()

X = stanscale.fit_transform(X)
y = stanscale.fit_transform(y.reshape(-1,1))



X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,
                                                    random_state=10)

from sklearn.svm import SVR

model = SVR(kernel='rbf')

model.fit(X_train,y_train)

y_pred = model.predict(X_test)
# y_test = stanscale.inverse_transform(y_test)
print(y_test)
print()
#y_pred = stanscale.inverse_transform(y_pred.reshape(-1,1))

print(y_pred)

print(r2_score(y_test, y_pred))
