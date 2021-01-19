# y = b0 + b1 * x + error
# b1 = ((n*Summ(xy)) - (Summ(x)*Summ(y))) / (n*Summ(x^2) - (Summ(x)^2))
# b0 = (Summ(y) - b1*Summ(x)) / n

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

    def summ_one_variable(self, data):
        # summation of all values of variable
        return sum(data)

    def summ_product_of_two_variables(self, x, y):
        product = [x*y for x,y in zip(x,y)]
        return sum(product)

    def square_values(self, data):
        # square all of values of a variable
        return [x*x for x in data]

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

