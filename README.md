# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages.

2. Print the present data and placement data and salary data.

3. Using logistic regression find the predicted values of accuracy confusion matrices.

4. Display the results.


```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: sriram E
RegisterNumber:  212223040207
*/
```
## Program & output:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv("/content/Placement_Data.csv")
df
```
![image](https://github.com/user-attachments/assets/dfbedcd2-01dd-40be-aae9-5280d670c67b)
```
df.info()
```
![image](https://github.com/user-attachments/assets/1a111b0a-a286-4a3c-b1df-cd7c5b7a7aa5)
```
df=df.drop('salary',axis=1)
df.info()
```
![image](https://github.com/user-attachments/assets/0db944a5-f572-4466-b6fb-a08be8a433fd)
```
df['gender']=df['gender'].astype('category')
df['ssc_b']=df['ssc_b'].astype('category')
df['hsc_b']=df['hsc_b'].astype('category')
df['degree_t']=df['degree_t'].astype("category")
df['workex']=df['workex'].astype('category')
df['specialisation']=df['specialisation'].astype('category')
df['status']=df['status'].astype('category')
df['hsc_s']=df['hsc_s'].astype('category')
df.dtypes
```
![image](https://github.com/user-attachments/assets/4c3dfa2c-5a4a-473e-af9e-bdc37489d82c)
```
df.info()
```
![image](https://github.com/user-attachments/assets/20599714-4000-45f5-b1dd-4c0398261e33)

```
df['gender']=df['gender'].cat.codes
df['ssc_b']=df['ssc_b'].cat.codes
df['hsc_b']=df['hsc_b'].cat.codes
df['degree_t']=df['degree_t'].cat.codes
df['workex']=df['workex'].cat.codes
df['specialisation']=df['specialisation'].cat.codes
df['status']=df['status'].cat.codes
df['hsc_s']=df['hsc_s'].cat.codes
df
```
![image](https://github.com/user-attachments/assets/d805dd00-cc1d-4268-bb3b-2870b142d951)
```
df.info()
```
![image](https://github.com/user-attachments/assets/3d3cecf5-3a90-4996-b9ad-2a17009c59c2)
```
x=df.iloc[:, :-1].values
y=df.iloc[:,-1].values
y
```
![image](https://github.com/user-attachments/assets/aec4336b-798b-497f-a6a5-f2939bdba528)
```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)
print(x_train.shape)
print(x_test.shape)
```
![image](https://github.com/user-attachments/assets/12e8e1b1-8e20-41a2-8c5f-c7c27e49290d)
```
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)

from sklearn.metrics import accuracy_score,confusion_matrix
print(accuracy_score(y_pred,y_test))
print(confusion_matrix(y_pred,y_test))
```
![image](https://github.com/user-attachments/assets/f6e64e5c-d12c-4dca-95e9-f3abe001e2f2)
```
# predict for new input
lr.predict([[0,87,0,95,0,2,8,0,0,1,5,6,5]])
lr.predict([[1,2,3,4,5,6,7,8,9,10,11,12,13]])
```
![image](https://github.com/user-attachments/assets/c9ea4851-bc71-4900-bdcc-4a851f0dda2c)

![image](https://github.com/user-attachments/assets/625b64d2-a805-462b-9e98-3ebffc8a15c1)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
