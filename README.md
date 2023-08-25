# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.
```
## Program:
```
Developed by: S.Kishore
RegisterNumber: 212222240050

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)

```

## Output:
1. df.head()
 
![image](https://github.com/Kishore2o/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118679883/8d05722d-7dda-4bad-b497-a61c01d6a7d4)

2. df.tail()

![image](https://github.com/Kishore2o/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118679883/87ea0a81-c817-4a06-bfcc-a0440b942651)

3. Array value of X

![2 2](https://github.com/Kishore2o/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118679883/aa54aae1-647b-45d6-ab39-a78ac810b628)

4. Array value of Y

![2 3](https://github.com/Kishore2o/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118679883/89d47a49-80d6-4a62-b9a1-cabe915762fd)

5. Values of Y prediction

![2 5](https://github.com/Kishore2o/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118679883/d890a8c1-4374-490a-beaa-cf8f4f29e359)

6. Array values of Y test

![2 6](https://github.com/Kishore2o/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118679883/0856f116-0304-4ff9-86cd-004eefd49a1a)

7. Training Set Graph

![2 7](https://github.com/Kishore2o/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118679883/bee1844d-df74-42b4-b45d-30c25be162b5)

8. Test Set Graph

![2 8](https://github.com/Kishore2o/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118679883/8125b72a-576f-4899-b9d2-c41cb39f2238)

9. Values of MSE, MAE and RMSE
   
![2 9](https://github.com/Kishore2o/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118679883/39a5a129-7a06-42bc-bd91-14da1e42b9f0)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
