import numpy as np
import pandas as pd

#https://www.kaggle.com/datasets/msjaiclub/2classclassification?resource=download
def get_data():
    df = pd.read_csv('../datasets/ex2data1.csv')
    return df


def get_structure():
    return get_data().columns