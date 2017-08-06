#coding:utf-8

from Network import Network

net =Network([2,3,1])
net.SGD(training_data=training_data,30,10,3.0,test_data=test_data)
print 'exit'
exit(0)

