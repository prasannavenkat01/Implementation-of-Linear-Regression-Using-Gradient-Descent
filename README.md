# Implementation of Linear Regression Using Gradient Descent
## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas, numpy and mathplotlib.pyplot.
2. Trace the best fit line and calculate the cost function.
3. Calculate the gradient descent and plot the graph for it.
4. Predict the profit for two population sizes.

## Program:
```
Program to implement the linear regression using gradient descent.
Developed by: DARIO G
RegisterNumber:  212222230027
```
```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.01,num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]
    theta=np.zeros(X.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        predictions=(X).dot(theta).reshape(-1,1)
        errors=(predictions-y).reshape(-1,1)
        theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
data=pd.read_csv('50_Startups.csv',header=None)
print(data.head())

X=(data.iloc[1:, :-2].values)
print(X)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
print(y)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X1_Scaled)
print(Y1_Scaled)

theta=linear_regression(X1_Scaled,Y1_Scaled)

new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}")

```

## Output:
### data
![image](https://github.com/prasannavenkat01/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/150702500/049d189a-3bb5-4524-bedf-aba4baf80af6)

### X values
![image](https://github.com/prasannavenkat01/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/150702500/c4d0ee95-2521-4e22-bb9e-ae4c59defb07)

### y values
![image](https://github.com/prasannavenkat01/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/150702500/5b431626-08c6-4f23-8292-9b0060dc1ef7)

###  X scaled 
![image](https://github.com/prasannavenkat01/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/150702500/e4ab7438-65b9-4f9c-b564-d2eaeaf7f5d3)

### y scaled
![image](https://github.com/prasannavenkat01/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/150702500/6a06ba8a-ac05-4674-ba1b-296fad18ae6e)

### Predicted value
![image](https://github.com/prasannavenkat01/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/150702500/d97bcf56-68bd-405f-8358-ab78e14104a8)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
