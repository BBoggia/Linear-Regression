from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

def main():
    print("Linear Regression Algorithm: ")
    pointCount = int(input("Number of points: "))
    print("Enter the amount of noise to be used in the point generation or skip for default (default: 75)")
    noise = input("noise: ")
    if noise == "":
        noise = 75
    else:
        noise = int(noise)
    x, y = datasets.make_regression(pointCount, noise = noise, n_features = 1, random_state = np.random.randint(0, 100))
    plt.figure(figsize=(8,8))
    plt.scatter(x, y, c=x, cmap=plt.cm.spring)
    regressor = regression_algo(pointCount, x, y)

    lineFitX = np.linspace(-3, 3, 100)
    lineFitY = regressor.slope*lineFitX + regressor.intercept
    plt.plot(lineFitX, lineFitY)

    plt.show()
    

class regression_algo():
    
    def __init__(self, point_count, x, y) -> None:
        self.point_count = point_count
        self.xList = x
        self.yList = y
        self.x_sum = x.sum()
        self.y_sum = y.sum()
        self.x_avg = self.x_sum / point_count
        self.y_avg = self.y_sum / point_count
        self.x_var = self.xVariance()
        self.co_var = self.coVariance()
        self.slope = self.co_var / self.x_var
        self.intercept = self.y_avg - self.slope * self.x_avg
        print("Slope: " + str(self.slope))
        print("Intercept: " + str(self.intercept)) 

    def xVariance(self):
        v = 0.0
        for i in range(self.point_count):
            v += (self.xList[i] - self.x_avg) ** 2
        return ((1 / (self.point_count - 1)) * v)

    def coVariance(self):
        v = 0
        for i in range(self.point_count):
            v += (self.xList[i] - self.x_avg) * (self.yList[i] - self.y_avg)
        return ((1 / (self.point_count - 1)) * v)
    
main()
