import matplotlib.pyplot as mpl
import numpy as np
from svm import calculus, kernel

ax = []
title_num = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
color = {'virginica': 'green', 'versicolor': 'red'}


def canvas(data):
    fig = mpl.figure()
    act_axes = [2, 3, 4, 6, 7, 8]
    for i in range(6):
        ax.append(fig.add_subplot(3, 3, act_axes[i]))
        ax[i].set_xlabel(data[0, title_num[i][0]], size=7)
        ax[i].set_ylabel(data[0, title_num[i][1]], size=7)


def scatter_plot(data, method):
    matrix_shape = {'svm': 4, 'knn': 5}
    for elem in data[1:]:
        for i in range(6):
            ax[i].scatter(
                float(elem[title_num[i][0]]),
                float(elem[title_num[i][1]]),
                c=color[elem[matrix_shape[method]]],
                s=7
                )
    mpl.show()


def color_mesh(data):
    width = 0.05
    init_data = data
    data = np.delete(np.transpose(data), [4, 5], axis=0).astype('float64')
    quality = data[4]
    data = np.delete(data, 4, axis=0)
    class_rate = []
    cmap = mpl.get_cmap('PiYG')
    ax_limits = [
        [min(data[0]), max(data[0])], [min(data[1]), max(data[1])],
        [min(data[2]), max(data[2])], [min(data[3]), max(data[3])]
    ]
    x_grid = []
    y_grid = []
    cell_x = np.zeros(6)
    cell_y = np.zeros(6)
    for i in range(6):
        cell_x[i] = (
            ax_limits[title_num[i][0]][1]-ax_limits[title_num[i][0]][0])*width
        cell_y[i] = (
            ax_limits[title_num[i][1]][1]-ax_limits[title_num[i][1]][0])*width
        x_grid.append(np.arange(
            ax_limits[title_num[i][0]][0],
            ax_limits[title_num[i][0]][1]+cell_x[i]*2, cell_x[i])
            )
        y_grid.append(np.arange(
            ax_limits[title_num[i][1]][0],
            ax_limits[title_num[i][1]][1]+cell_y[i]*2, cell_y[i])
            )
    for i in range(6):
        indices_x = (
            (data[title_num[i][0]]-min(data[title_num[i][0]]))
            / cell_x[i]).astype(int)
        indices_y = (
            (data[title_num[i][1]]-min(data[title_num[i][1]]))
            / cell_y[i]).astype(int)
        class_rate.append(np.zeros((int(1/width)+2, int(1/width)+2)))
        for point in range(data.shape[1]-1):
            class_rate[i][indices_y[point], indices_x[point]] = quality[point]
    ind = 0
    for i in range(3):
        for axes in class_rate:
            for row in range(1, axes.shape[0]-1):
                for col in range(1, axes.shape[1]-1):
                    if axes[row, col] == 0:
                        class_rate[ind][row, col] = (
                            axes[row+1, col]
                            + axes[row-1, col]
                            + axes[row, col+1]
                            + axes[row, col-1]
                            )/2
            ind += 1
        ind = 0
    for i in range(6):
        ax[i].pcolormesh(
            x_grid[i],
            y_grid[i],
            class_rate[i],
            cmap=cmap,
            alpha=0.6
            )
    scatter_plot(init_data, 'knn')
    mpl.show()
    return data


def svm_plot(data):
    w_real, w0_real, err_real = calculus(data, dim=4) 
    x = data[data.shape[0]//3*2:, :4].astype('float64')
    y = data[data.shape[0]//3*2:, 4]
    x_t = np.delete(np.transpose(x), [4, 5], axis=0).astype('float64')
    ax_limits = [
        [min(x_t[0]), max(x_t[0])], [min(x_t[1]), max(x_t[1])],
        [min(x_t[2]), max(x_t[2])], [min(x_t[3]), max(x_t[3])]
    ]
    for axes in range(6):
        ax_data = np.hstack(
            (np.reshape(x[:, title_num[axes][0]], (x.shape[0], 1)),
                np.reshape(x[:, title_num[axes][1]], (x.shape[0], 1)),
                np.reshape(y, (y.shape[0], 1)))
            )
        w, w0, err = calculus(ax_data, dim=2)
        x_seq = np.linspace(ax_limits[title_num[axes][0]][0],
                            ax_limits[title_num[axes][0]][1])
        ax[axes].plot(x_seq, -x_seq*(w[0]/w[1])+w0/w[1])
    scatter_plot(data[data.shape[0]//3*2:], 'svm')
    print('There are not actual dividing lines')
    print('Actual equation of dividing surface is:')
    print(str(w_real[0])+'*x_1 + '
                        + str(w_real[1])+'*x_2 + '
                        + str(w_real[2])+'*x_3 + '
                        + str(w_real[3])+'*x_4 + '
                        + ' - ' + str(w0_real) + '= 0')
    print()
    print('Mistakes rate on production set is: ' + str(err_real) + '%')
    mpl.show()
