# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.

2.Upload and read the dataset.

3.Check for any null values using the isnull() function.

4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.

5.Find the accuracy of the model and predict the required values by importing the required module from sklearn. 
 

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: BHUVANESHWARI S
RegisterNumber: 212222220008  
*/
```

import pandas as pd
data = pd.read_csv("Employee (1).csv")
data.head()

data.info()
data.isnull().sum()
data['left'].value_counts()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data['salary'] = le.fit_transform(data['salary'])
data.head()

x=data[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','salary']]
x.head()

y=data['left']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state =100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion='entropy')
dt.fit(x_train,y_train)
y_predict=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_predict)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])****

## Output:

# DATA HEAD:
![image](https://github.com/user-attachments/assets/86b8377c-40a2-4b80-b977-89504d03bf79)

# DATASET INFO:
![image](https://github.com/user-attachments/assets/235bd985-0c13-474b-b8fc-ff385f4d047a)

# NULL DATASET:
![image](https://github.com/user-attachments/assets/3061370b-f8a7-4d04-ab4c-698653302b88)

# VALUES COUNT IN LEFT COLUMN:
![image](https://github.com/user-attachments/assets/89cdd522-18ef-4267-8c3f-87003533e57f)

# DATASET TRANSFORMED HEAD:
![image](https://github.com/user-attachments/assets/3582d99d-8338-43a4-bf8d-f4a61f10e0d6)

# X.HEAD:
![image](https://github.com/user-attachments/assets/649f0bf6-916e-43a1-8571-f6d67f5c89a2)

# CCURACY:
![image](https://github.com/user-attachments/assets/2d8223bc-608b-48a3-b23c-ef40d768bd92)

# DATA PREDICTION:

![image](https://github.com/user-attachments/assets/98635c41-cb15-4752-b848-02bfd9765f23)





## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
