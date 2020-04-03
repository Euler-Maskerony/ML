import numpy as np
from scipy.optimize import minimize


def fromfunc(func, shape, **kwargs):
    arr = np.zeros(shape)

    def arr_creator(shape, *args):
        shape = list(shape)
        count = shape.pop()
        for i in range(count):
            if shape != []:
                arr_creator(shape, i, args)
            else:
                args = [i] + list(args)
                if args[:].pop() == ():
                    args.pop()
                args_t = tuple(args)
                arr[args_t] = func(*args, **kwargs)
                args.pop(0)
        if args:
            args.pop(0)

    arr_creator(shape)
    return arr


def cross_validation(data, start, stop, step):
    min_err = 100
    c = None
    for param in np.arange(start, stop, step):
        err = calculus(data, param)[3]
        if err < min_err:
            min_err, c = err, param
        print('Parameter: ' + str(param))
        print('Errors: ' + str(err))
    print('Cross validation has been completed succesfully')
    return err, c


def kernel(a, b):
    return np.dot(a, b)


def jac_creator(l, x, y, i):
    first = 0
    for j in range(l.shape[0]):
        first += l[j]*y[i]*y[j]*kernel(x[i], x[j])
    return first-1


def func(l, *args):
    frst = (-1)*np.sum(l)
    scnd = 0
    for i in range(l.shape[0]):
        for j in range(l.shape[0]):
            scnd += l[i]*l[j]*args[1][i]*args[1][j]*kernel(
                args[0][i], args[0][j])
    func = frst+0.5*scnd
    return func


def func_jac(l, *args):
    jac = np.zeros(l.shape[0])
    for i in range(l.shape[0]):
        jac[i] = jac_creator(l, args[0], args[1], i)
    return jac


def func_eq_constr(l, y):
    res = 0
    for i in range(l.shape[0]):
        res += l[i]*y[i]
    return res


def jac_eq_constr(*axis, **kwargs):
    return kwargs['y'][axis[0]]


def func_ineq_constr(*axis, **kwargs):
    size = kwargs['var'].shape[0]
    if axis[0] <= (size-1):
        return kwargs['var'][axis[0]]
    else:
        return kwargs['c']-kwargs['var'][axis[0]-size]


def jac_ineq_constr(*axis, **kwargs):
    size = kwargs['var'].shape[0]
    if axis[0] <= (size-1):
        if axis[0] == axis[1]:
            return 1
        else:
            return 0
    else:
        if (axis[0]-size) == axis[1]:
            return -1
        else:
            return 0


def calculus(data, c=5, dim=4):
    w = np.zeros(dim)
    w0 = 0
    non_zero = []
    iris_type = {'versicolor': -1, 'virginica': 1}
    data = np.delete(data, 0, axis=0)
    x_train = data[:data.shape[0]//3*2, :dim].astype('float64')
    x_control = data[data.shape[0]//3*2:, :dim].astype('float64')
    y_train = np.zeros(x_train.shape[0])
    y_control = np.zeros(x_control.shape[0])
    for i in range(y_train.shape[0]):
        y_train[i] = iris_type[data[i, dim]]
    for i in range(y_control.shape[0]):
        y_control[i] = iris_type[data[i+data.shape[0]//3*2, dim]]
    ineq_cons = {
        'type': 'ineq',
        'fun': lambda l: fromfunc(
            func_ineq_constr,
            (l.shape[0]*2,),
            var=l,
            c=c
        ),
        'jac': lambda l: fromfunc(
            jac_ineq_constr,
            (l.shape[0]*2, l.shape[0]),
            var=l
        )
    }
    eq_cons = {
        'type': 'eq',
        'fun': lambda l: np.array([
            func_eq_constr(l, y_train)
        ]),
        'jac': lambda l: fromfunc(
            jac_eq_constr,
            (l.shape[0],),
            y=y_train
        )
    }
    parameters = minimize(
            func,
            np.ones(x_train.shape[0]),
            args=(x_train, y_train),
            jac=func_jac,
            method='SLSQP',
            constraints=[ineq_cons, eq_cons],
            options={'ftol': 1e-9, 'disp': False, 'maxiter': 1000},
        )
    var_array = np.around(parameters.x, 4)
    for i in range(var_array.shape[0]):
        w += x_train[i]*y_train[i]*var_array[i]
        if round(var_array[i], 2) != 0:
            non_zero.append(i)
    for el in non_zero:
        w0 += (kernel(w, x_train[el]) - y_train[el]) / len(non_zero)
    w0 = round(w0, 5)
    w = np.around(w, 5)
    err = 0
    for obj in range(x_control.shape[0]):
        class_ind = np.sign(kernel(w, x_control[obj]) - w0)
        if class_ind != y_control[obj]:
            err += 1
    err = round(err/x_control.shape[0]*100, 3)
    return w, w0, err
