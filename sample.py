from simple_linear_regression import LinearRegression

x = [1,2,3,4,5,6,7,8]
y = [30,45,51,57,60,65,70,71]
model = LinearRegression(x, y)
print("Simple linear regression model")
print(f"b0: {model.b0}, b1: {model.b1}")
print(f"For the x value 10, the model predicts {model.predict(10)}")
