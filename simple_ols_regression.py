import numpy as np

class LinearRegression():
    def __init__(self, x, y):
        self.y = y

        # add intercept to x matrix
        intercept_ones = np.ones((x.shape[0],1))
        self.x = np.concatenate((intercept_ones, x.reshape(-1,1)), axis=1)

        self.yhat = []
    
    def fit(self):
        # (X'X)^(-1)X'Y to calculate OLS coefficient estimates
        self.b = np.dot(np.linalg.inv(np.dot(self.x.T, self.x)), np.dot(self.x.T, self.y))
        self.yhat = np.dot(self.x, self.b)
    
    def predict(self, input):
        # predict y value for given x
        input_array = np.array([1, input])
        return np.dot(input_array, self.b)
    
    def calc_t_stat(self):
        # calculate standard error of the slope
        # se = sqrt(Summ((y-yhat)**2)/n-2) / sqrt(Summ((xi - xbar)**2))
        # t = b1/se(b1)
        num = 0
        for yi, yhat in zip(self.y, self.yhats):
            num += (yi - yhat)**2
        num = num / (self.n - 2)
        num = num**0.5

        den = 0
        xbar = sum(self.x)/len(self.x)
        for xi in self.x:
            den += (xi - xbar)**2
        den = den**0.5
        
        se = num / den

        t = self.b1/se

        return t
    
    def calc_r2(self):
        # SSE/SST
        # SSE = Sum((yihat - ybar)**2)
        # SST = Sum((yi - ybar)**2)
        ybar = sum(self.y)/len(self.y)
        sse = 0
        for yihat in self.yhats:
            sse += (yihat - ybar)**2
        sst = 0
        for yi in self.y:
            sst += (yi - ybar)**2
        
        return sse/sst


