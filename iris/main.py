import csv
import numpy as np
import plot
from knn import shuffle, knn

count = -1
data_l = []
with open('data/iris.csv', newline='') as file:
    reader = csv.reader(file)
    for iris in reader:
        count += 1
        if count <= 51 and count > 0:
            continue
        data_l.append(iris)
data = np.array(data_l)
if __name__ == "__main__":
    plot.canvas(data)
    data = shuffle(data)
    #plot.color_mesh(knn(data))
    plot.svm_plot(data)
