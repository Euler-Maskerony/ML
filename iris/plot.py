import matplotlib.pyplot as mpl
import numpy as np

ax = []
title_num = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)] 
color = {'setosa': 'yellow', 'versicolor': 'red', 'virginica': 'green'}


def scatter_plot(data):
    fig = mpl.figure()
    for i in range(1, 7):
        ax.append(fig.add_subplot(3, 2, i))
        ax[i-1].set_xlabel(data[0, title_num[i-1][0]], size=7)
        ax[i-1].set_ylabel(data[0, title_num[i-1][1]], size=7)
        for elem in data[1:]:
            mpl.scatter(float(elem[title_num[i-1][0]]), float(elem[title_num[i-1][1]]), c=color[elem[4]], s=6)
    mpl.show()
        

