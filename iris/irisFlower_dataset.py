import csv
import numpy as np
from plot import scatter_plot


data_l = []
with open('D://MyOwn/programs/iris/data/iris.csv', newline='') as file: #Своё напиши лол)
    reader = csv.reader(file)
    for iris in reader:
        data_l.append(iris)
data = np.array(data_l)

if __name__ == "__main__":
    scatter_plot(data)