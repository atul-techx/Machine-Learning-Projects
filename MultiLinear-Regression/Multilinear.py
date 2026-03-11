import numpy as np
import matplotlib .pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

x=np.array([1,2,3,4,5]).reshape(-1,1)
y=np.array([1,4,9,16,25])

linear_model=LinearRegression()
linear_model.fit(x,y)
linear_pred=linear_model.predict(x)

poly_model=make_pipeline(PolynomialFeatures(degree=2)),LinearRegression(())