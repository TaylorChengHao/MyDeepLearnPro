#coding:utf-8
import numpy as np
import random

class Network(object):

    def __init__(self,sizes):
        # 网络层数
        self.num_layers=len(sizes)
        self.sizes=sizes
        # 随机初始化
        # 偏置，均值为0，标准差为1的高斯分布（正态分布）,randn(a,b)参数a是矩阵行数，b是列数
        self.biases=[np.random.randn(y,1) for y in sizes[1:]]
        # 权重
        # zip将两个list中的同下标元素，两两为一个tuple，多个tuple组成一个list
        self.weights=[np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:])]

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
                # eta是学习速率,对每一个mini_batch应用一次梯度下降
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
    def update_mini_batch(self,mini_batch,eta):
        # 得到偏置和权重的零矩阵
        nabla_b=[np.zeros(b.shape) for b in self.biases]
        nabla_w=[np.zeros(w.shape) for w in self.weights]

        for x,y in mini_batch:
            # 反向传播算法
            delta_nabla_b,delta_nabla_w=self.backprop(x,y)
            # 更新权重和偏置
            nabla_b=[nb+dnb for nb,dnb in zip(nabla_b,delta_nabla_b)]
            nabla_w=[nw+dnw for nw,dnw in zip(nabla_w,delta_nabla_w)]
            self.weights=[w-(eta/len(mini_batch))*nw for w,nw in zip(self.weights,nabla_w)]
            self.biases=[b-(eta/len(mini_batch))*nb for b,nb in zip(self.biases,nabla_b)]

    def backprop(self,x,y):
        # x是训练数据的输入，y是训练数据的输出

        # 先创建形状一样的零矩阵
        nable_b=[np.zeros(b.shape) for b in self.biases]
        nable_w=[np.zeros(w.shape) for w in self.weights]
        #feedforward前向传播 输入层---->隐含层
        activation=x
        activations=[x]
        zs=[] #记录输出层误差
        for b,w in zip(self.biases,self.weights):
            # 点积，将传入的x和权重点乘加上偏置
            # 将上面的结果传入S型函数，
            # S型函数返回结果就是下一个神经元的输入值，
            # 和之前的输入值（x）一起存在activations中
            z=np.dot(w,activation)+b
            zs.append(z) #暂存带权输入值
            activation=sigmoid(z)
            activations.append(activation)

        # backward pass 输出层误差
        delta=self.cost_derivative(activations[-1],y)*sigmod_prime(zs[-1])
        nable_b[-1]=delta
        nable_w[-1]=np.dot(delta,activations[-2].transpose())

        # 这是为了将更新好的梯度
        for l in xrange(2,self.num_layers):
            z=zs[-l]
            sp=sigmod_prime(z)
            # 这里是在讲输出层误差和训练的输入值的乘积
            delta=np.dot(self.weights[-l+1].transpose(),delta)*sp
            nable_b[-l]=delta
            nable_w[-l]=np.dot(delta,activations[-l-1].transpose())

        return (nable_b,nable_w)

    def evaluate(self,test_data):
        test_results=[(np.argmax(self.feedforward(x)),y) for(x,y) in test_data]
        return sum(int(x==y) for (x,y) in test_results)

    def cost_derivative(self,output_activations,y):
        return (output_activations-y)

# S型函数
def sigmoid(z):
    return  1.0/(1.0+np.exp(-z))

def sigmod_prime(z):
    return sigmoid(z)*(1-sigmoid(z))
