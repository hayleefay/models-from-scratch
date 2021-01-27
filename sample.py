import numpy as np
from simple_ols_regression import LinearRegression

x = np.array([1,2,3,4,5,6,7,8])
y = np.array([30,45,51,57,60,65,70,71])
model = LinearRegression(x, y)
model.fit()
print("Simple linear regression model")
print(f"b0: {model.b[0]}, b1: {model.b[1]}")
print(f"For the x value 10, the model predicts {model.predict(10)}")
# print(model.calc_t_stat())
# print(model.calc_r2())

# compare to statsmodel
# import statsmodels.api as sm
# mod = sm.OLS(y, sm.add_constant(x))
# res = mod.fit()
# print(res.summary())

