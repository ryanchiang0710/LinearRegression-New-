import numpy as np
import pandas as pd
url = "https://raw.githubusercontent.com/GrandmaCan/ML/main/Resgression/Salary_Data.csv"
data = pd.read_csv(url)
data_x = data["YearsExperience"]
data_y = data["Salary"]
num_x = len(data_x)
num_y = len(data_y)

def average_x_y(data_x,data_y):
    average_x = np.sum(data_x)/num_x
    average_y = np.sum(data_y)/num_y
    return average_x,average_y

average_x,average_y = average_x_y(data_x,data_y)
print("The average of x,y: ",average_x,average_y)

def sigma_x_y(data_x,average_x,data_y,average_y):
    sigma_x = (np.sum((data_x-average_x)**2))**0.5
    sigma_y = (np.sum((data_y-average_y)**2))**0.5
    return sigma_x,sigma_y
sigma_x,sigma_y = sigma_x_y(data_x,average_x,data_y,average_y)
print("Sigma_x and sigma_y: ",sigma_x,sigma_y)
pre_x = np.sum((data_x-average_x)**2)
pre_y = np.sum((data_y-average_y)**2)
pre_xy = np.sum((data_x-average_x)*(data_y-average_y))

def rxy(pre_x,pre_y,pre_xy):
    rxy = pre_xy/((pre_x*pre_y)**0.5)
    return rxy
rxy = rxy(pre_x,pre_y,pre_xy)
print("The correlation of x,y is: ",rxy)

def model(average_x,average_y,sigma_x,sigma_y,rxy):
    x = input("Enter a data for prediction: ")
    y = (float(x)-average_x)*rxy*(sigma_y/sigma_y)+average_y
    return y
y = model(average_x,average_y,sigma_x,sigma_y,rxy)
print("The predicted value is: ",y)