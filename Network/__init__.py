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

    # S型函数
    def sigmoid(self,z):
        return  1.0/(1.0+np.exp(-z))

    # 前馈，针对每一层应用S型方程
    def feedforward(self,a):
        # zip()函数将权重和偏置转化为元组对列表
        # 意在将每一个偏置和多个权重（biases的list中的、weights中各list中同样下标位置的元素）
        # 代入S型函数
        for b,w in zip(self.biases,self.weights):
            #将权重和输入值的点集外加偏置
            a=self.sigmoid(np.dot(w,a)+b)
        return a

    # 随机梯度下降
    def SGD(self,training_data,epochs,mini_batch_size,eta,test_data=None):
        if test_data:n_test=len(test_data)
        n=len(training_data)
        # xrange相比于range,range会直接返回一个list，
        # xrange则不会直接生成一个list，而是每次调用返回其中的一个值
        # xrange性能更好
        for j in xrange(epochs):
            # training_data是一个（x,y）元组的列表，代表训练输入和期望输出
            # epochs是迭代期数量，mini_batch_size采样的小批量数据的大小
            random.shuffle(training_data)
            # 将training_data分为多个batch
            mini_batches=[training_data[k:k+mini_batch_size]
                          for k in xrange(0,n,mini_batch_size)]
            for mini_batch in mini_batches:
                # eta是学习速率
                self.update_mini_batch(mini_batch,eta)
            if test_data:
                print "Epoch {0}:{1}/{2}".format(j,self.evaluate(test_data),n_test)
            else:
                print "Epoch {0} complete".format(j)

    #代码如下⼯作。在每个迭代期，它⾸先随机地将训练数据打乱，然后将它分成多个适当⼤
    # ⼩的⼩批量数据。这是⼀个简单的从训练数据的随机采样⽅法。然后对于每⼀个 mini_batch
    # 我们应⽤⼀次梯度下降。这是通过代码 self.update_mini_batch(mini_batch, eta) 完成的，它仅
    # 仅使⽤ mini_batch 中的训练数据，根据单次梯度下降的迭代更新⽹络的权重和偏置。这是
    # update_mini_batch ⽅法的代码：