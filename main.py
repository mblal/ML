import pandas as pd
import numpy as np
from cost_function import cost_function
from gradient_descent import gradient_descent
from compute_gradient_descent import compute_gradient
df = pd.read_csv('datasets/Student_Performance.csv')
df['Extracurricular Activities'] = df['Extracurricular Activities'].map({"Yes":1, "No":0})
target = 'Performance Index'

x_train = df.values[:,0:5]
y_train = df.values[:,-1]

'''
    let's suppose that
    w1 = 1
    w2 = 1
    w3 = 1
    w4 = 1
    w5 = 1
    b  = 10
'''
w = np.array([1, 1, 1, 1, 1])
b = 10
iterations = 10000
#print(cost_function(x_train, y_train, w, b))
#print(gradient_descent(x_train, y_train))
compute_gradient(x_train, y_train, iterations)
