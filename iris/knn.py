import numpy as np
import irisFlower_dataset as main


def kernel(dist, a):
    ker = a*(1-dist**2)
    if ker > 0:
        return ker
    else:
        return 0


def sort_adv(mas):
    for i in range(len(mas)):
        for j in range(i, len(mas)):
            if mas[i, 0] > mas[j, 0]:
                c = np.copy(mas[i])
                mas[i] = np.copy(mas[j])
                mas[j] = np.copy(c)
    return mas


def shuffle(data):
    for iteration in range(10000):
        i = np.random.randint(1, 150)
        j = np.random.randint(1, 150)
        c = np.copy(data[i])
        data[i] = np.copy(data[j])
        data[j] = np.copy(c)
    return data


def argmax(control_obj, a=1):
    quality = np.zeros(3)
    for dist in control_obj:
        if dist[0] == 0:
            break
        quality[int(round(dist[1]))] += kernel(dist[0], a)
    if quality[0] == np.max(quality):
        return 'setosa'
    elif quality[1] == np.max(quality):
        return 'versicolor'
    elif quality[2] == np.max(quality):
        return 'virginica'

    
def cross_validation(data):
    a = 10
    first = knn(shuffle(data), a)
    while a > 0:
        a -= 0.1
        for i in range(30):
            last = knn(data, a)
            if last < first:
                min_a = last
    return min_a


def knn(data, a=1):
    row = 0
    col = 0
    ratio = 0
    iris_type = {'virginica': 2, 'versicolor': 1, 'setosa': 0}
    data = shuffle(data)
    control = data[1:data.shape[0]//3]
    learning = data[data.shape[0]//3+1:data.shape[0]]
    dist = np.zeros((control.shape[0], learning.shape[0], 2))
    control_classes = np.zeros((49, 1))
    control_classes = control_classes.astype(str)
    for obj in control:
        for target in learning:
            dist[row, col] = [(float(obj[0])-float(target[0]))**2 + (float(obj[1])-float(target[1]))**2 + (float(obj[2])-float(target[2]))**2 + (float(obj[3])-float(target[3]))**2, iris_type[target[4]]]
            col += 1
        dist[row] = sort_adv(dist[row])
        control_classes[row, 0] = argmax(dist[row], a)
        row += 1
        col = 0
    control = np.hstack((control, control_classes))
    for obj in control:
        if obj[4] != obj[5]:
            ratio += 1
        ratio = ratio/49
    print(control)
    print(ratio)
    return ratio


a = 1
if input('cv?') == 'y':
    a = cross_validation(main.data)
knn(main.data, a)
