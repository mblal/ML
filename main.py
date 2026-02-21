import numpy as np
import matplotlib.pyplot as plt
from cost_function import cost_function
from gradient_descent import gradient_descent
from prediction_model import predict
from pathlib import Path
import kagglehub
import shutil
import os
import pandas as pd
'''src = kagglehub.dataset_download("abhishek14398/salary-dataset-simple-linear-regression")

dataset_path = Path(src)
file_path = next(dataset_path.glob("*"))
dest = './datasets'

if not os.path.exists(dest + file_path.name):
    shutil.move(str(file_path), dest)'''

file_path = './datasets/Salary_dataset.csv'
df = pd.read_csv(file_path, index_col=0)

x_train = np.array(df.iloc[:,0])
y = np.array(df.iloc[:,1])

'''#Let's suppose here that w = 9000 and b=35000
w = 9000
b = 30000
result  = cost_function(x_train, y, w, b)'''

_w, _b, effort = gradient_descent(x_train, y)

print(predict(25, _w, _b))