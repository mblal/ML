import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cost_function import cost_function
from gradient_descent import gradient_descent
from compute_gradient_descent import compute_gradient
from z_score_normalization import z_score_normalization

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
w = np.array([0, 0, 0, 0, 0])
b = 0
iterations = 100
#print(cost_function(x_train, y_train, w, b))
#print(gradient_descent(x_train, y_train))
x_nom, mu, sigma = z_score_normalization(x_train)
_w, _b, history = compute_gradient(x_nom, y_train, iterations)

plt.scatter(range(iterations), history)
plt.show()
