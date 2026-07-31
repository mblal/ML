import numpy as np
import time
from utils import *
from sigmoid import *
from compute_gradient import *
from gradient_descente import *

raw_df = get_data()

x_train = np.array(raw_df.iloc[:,[0,1]])
y_train = np.array(raw_df.iloc[:,2])
m, n = x_train.shape

#w = np.zeros(n)
w = np.array([0, 0])
b = 0

#x_train_for_test = [[40, 60],[80, 80],[90, 65]]
#y_train_for_test = [0, 1, 0]
#x_train_for_test = np.array(x_train_for_test)
#y_train_for_test = np.array(y_train_for_test)


#z = np.dot(x_train, w) + b
#cost = compute_cost(x_train, y_train, w, b)
#print(cost)


''' Lets measure some performance here '''
debut = time.perf_counter()
dj_w, dj_b = compute_gradient(x_train, y_train, w, b)
fin = time.perf_counter()
print(f"Temps : {fin - debut:.4f} secondes")
print(dj_w, dj_b)

debut = time.perf_counter()
dj_w, dj_b = compute_gradient_v2(x_train, y_train, w, b)
fin = time.perf_counter()
print(f"Temps : {fin - debut:.4f} secondes")
print(dj_w, dj_b)

weights, bias = gradient_descent(x_train, y_train, w, b, 0.001, 10000000)




