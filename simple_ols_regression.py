class LinearRegression():
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.n = self.calculate_n(self.x)
        self.summ_xy = self.summ_product_of_two_variables(self.x, self.y)
        self.summ_x = self.summ_one_variable(self.x)
        self.summ_y = self.summ_one_variable(self.y)
        self.summ_xx = self.summ_product_of_two_variables(self.x, self.x)

        self.b1 = self.calculate_b1()
        self.b0 = self.calculate_b0()
        self.yhats = [self.predict(obs) for obs in self.x]

    def summ_one_variable(self, data):
        # summation of all values of variable
        return sum(data)

    def summ_product_of_two_variables(self, m, p):
        product = [m*p for m,p in zip(m,p)]
        return sum(product)

    def square_values(self, data):
        # square all of values of a variable
        return [m*m for m in data]

    def calculate_n(self, data):
        # count number of observations in variable
        return len(data)

    def calculate_b1(self):
        # b1 = ((n*Summ(xy)) - (Summ(x)*Summ(y))) / (n*Summ(x^2) - (Summ(x)^2))
        b1_num = (self.n*self.summ_xy) - (self.summ_x*self.summ_y)
        b1_den = (self.n*self.summ_xx) - (self.summ_x*self.summ_x)
        b1 = b1_num / b1_den
        return b1
    
    def calculate_b0(self):
        # b0 = (Summ(y) - b1*Summ(x)) / n
        b0 = (self.summ_y - (self.b1*self.summ_x)) / self.n
        return b0
    
    def predict(self, input):
        # predict y value for given x
        return self.b0 + self.b1*input
    
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


