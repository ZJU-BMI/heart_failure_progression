# encoding=utf-8-sig
import numpy as np
import GPyOpt
from matplotlib import pyplot as plt
import datetime
import random


def test_function_1d(x):
    return x**2 - 20*x + 20


def test_function_7d(x):
    y = x[0][0]**2 + x[0][1]**2 + x[0][2]**2 + x[0][3]**2 + x[0][4]**2 + x[0][5]**2 + x[0][6]**2
    return y


bayesian_opt_flag = False
iter_loss = []
if not bayesian_opt_flag:
    # random search
    minimum_f_x = 1000000
    for i in range(500000):
        x = [[random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1),
              random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1),
              random.uniform(-1, 1)]]
        y = test_function_7d(x)
        if y < minimum_f_x:
            minimum_f_x = y
        iter_loss.append([i, minimum_f_x])
    print('minimum loss {}'.format(minimum_f_x))

else:
    bounds = [{'name': 'var_1', 'type': 'continuous', 'domain': (-1, 1)},
              {'name': 'var_2', 'type': 'continuous', 'domain': (-1, 1)},
              {'name': 'var_3', 'type': 'continuous', 'domain': (-1, 1)},
              {'name': 'var_4', 'type': 'continuous', 'domain': (-1, 1)},
              {'name': 'var_5', 'type': 'continuous', 'domain': (-1, 1)},
              {'name': 'var_6', 'type': 'continuous', 'domain': (-1, 1)},
              {'name': 'var_7', 'type': 'continuous', 'domain': (-1, 1)}]
    max_iter = 50
    case_7d = GPyOpt.methods.BayesianOptimization(test_function_7d, bounds)

    start = datetime.datetime.now()
    case_7d.run_optimization(max_iter)
    time_elapse = (datetime.datetime.now() - start).seconds
    print('iter {}, optimum x: {}, optimum f(x): {}, total time cost {} seconds'.
          format(max_iter, case_7d.x_opt, case_7d.fx_opt, time_elapse))
