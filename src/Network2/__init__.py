# This block codes is for test.py
# import mnist_loader
# training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
# import network2
# net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
# net.large_weight_initializer()
# net.SGD(training_data, 30, 10, 0.5, evaluation_data=test_data,monitor_evaluation_accuracy=True)

import json
import  random
import sys

import numpy as np
# 使用交叉熵来作为代价函数

# 交叉熵代价函数的定义
class QuadraticCost(object):

    # 求矩阵范式的平方
    @staticmethod
    def fn(a,y):
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z,a,y):
        return (a-y)*sigmoid_prime(z)

class CrossEntropyCost(object):

    @staticmethod
    def fn(a,y):
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(z,a,y):
        return (a-y)

##Main Network Class
class Network(object):

    def __init__(self,sizes,cost=CrossEntropyCost):
        self.num_layers=len(sizes)

