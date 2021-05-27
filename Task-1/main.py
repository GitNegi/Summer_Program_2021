import pandas as pd
import numpy as np

dataset = pd.read_csv("/home/salarydata.csv")

X = dataset['YearsExperience'].values.reshape(-1,1)
y = dataset['Salary']

from sklearn.linear_model import LinearRegression
brain = LinearRegression()
brain.fit(X,y)
x = brain.predict([[float(input("Enter Years of Experience :-> "))]])
print(x[0])
