import numpy as np


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
        i = np.random.randint(1, 100)
        j = np.random.randint(1, 100)
        c = np.copy(data[i])
        data[i] = np.copy(data[j])
        data[j] = np.copy(c)
    return data


def argmax(control_obj, a=1):
    quality = np.zeros(2)
    for dist in control_obj:
        if dist[0] == 0:
            break
        quality[int(round(dist[1]))] += kernel(dist[0], a)
    if quality[0] == np.max(quality):
        return 'virginica', quality[0]-quality[1]
    elif quality[1] == np.max(quality):
        return 'versicolor', quality[0]-quality[1]


def cross_validation(data):
    a = 10
    min_qual = 10
    qual = 0
    while a > 0:
        for i in range(30):
            qual += knn(data, a)
        if qual < min_qual:
            min_a = a
            min_qual = qual
        a -= 1
        qual = 0
        print(min_a)
    return min_a


def knn(data, a=8):
    row = 0
    col = 0
    ratio = 0
    iris_type = {'versicolor': 1, 'virginica': 0}
    data = shuffle(data)
    control = data[1:data.shape[0]//3]
    learning = data[data.shape[0]//3+1:data.shape[0]]
    dist = np.zeros((control.shape[0], learning.shape[0], 2))
    control_classes = np.zeros((data.shape[0]//3-1, 1))
    control_classes = control_classes.astype(str)
    rate = np.zeros((data.shape[0]//3-1, 1))
    for obj in control:
        for target in learning:
            dist[row, col] = [
                (float(obj[0])-float(target[0]))**2
                + (float(obj[1])-float(target[1]))**2
                + (float(obj[2])-float(target[2]))**2
                + (float(obj[3])-float(target[3]))**2,
                iris_type[target[4]]]
            col += 1
        dist[row] = sort_adv(dist[row])
        control_classes[row, 0], rate[row, 0] = argmax(dist[row], a)[0], argmax(dist[row], a)[1]
        row += 1
        col = 0
    row = 0
    for i in rate:
        if i > 0:
            rate[row, 0] = np.round(i / max(rate), 4)
        else:
            rate[row, 0] = np.round(i / min(rate), 4) * (-1)
        row += 1
    control = np.hstack((control, control_classes, rate))
    for obj in control:
        if obj[4] != obj[5]:
            ratio += 1
        ratio = ratio/(data.shape[0]//3-1)
    print(control)
    print(ratio)
    return control
