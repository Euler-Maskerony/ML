import csv
import numpy as np
import plot

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
    #plot.color_mesh(knn(data))
    data = shuffle(data)
    plot.svm_plot(data)
