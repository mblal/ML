from utils import *
import matplotlib.pyplot as plt

raw_df = get_data()


fig, axes = plt.subplots()
axes.scatter((raw_df[raw_df['label'] == 0]).iloc[:,0],
             (raw_df[raw_df['label'] == 0]).iloc[:,1], marker='o', color='b')

axes.scatter((raw_df[raw_df['label'] == 1]).iloc[:,0],
             (raw_df[raw_df['label'] == 1]).iloc[:,1], marker='*', color='r')
plt.title('Illustrate decision boundry')
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.show()