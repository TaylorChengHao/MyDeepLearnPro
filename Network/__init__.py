#coding:utf-8
import numpy as np
import random

class Network(object):

    def __init__(self,sizes):
        # 网络层数
        self.num_layers=len(sizes)
        self.sizes=sizes
        # 随机初始化
        # 偏置，均值为0，标准差为1的高斯分布（正态分布）
        self.biases=[np.random.randn(y,1) for y in sizes[1:]]
        # 权重
        self.weights=[np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:])]

    def sigmoid(self,z):
        return  1.0/(1.0+np.exp(-z))

    def feedforward(self,a):
        for b,w in zip(self.biases,self.weights):
            a=self.sigmoid(np.dot(w,a)+b)
        return a

    def SGD(self,training_data,epochs,mini_batch_size,eta,test_data=None):
        if test_data:n_test=len(test_data)
        n=len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches=[training_data[k:k+mini_batch_size]
                          for k in xrange(0,n,mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch,eta)
            if test_data:
                print "Epoch {0}:{1}/{2}".format(j,self.evaluate(test_data),n_test)
            else:
                print "Epoch {0} complete".format(j)
