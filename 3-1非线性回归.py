# -*- coding:utf-8 -*- 
'''
    @Author: Tomas Wu 
    @Date: 2019-03-25 23:45:06 
    @Desc: 
'''
import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt

# 生成200个随机点
xdata=np.linspace(-0.5,0.5,200)[:,np.newaxis]
noise=np.random.normal(0,0.02,xdata.shape)
ydata=np.square(xdata)+noise

# 定义两个placeholder
x=tf.placeholder(tf.float32,[None,1])
y=tf.placeholder(tf.float32,[None,1])
# 构建神经网络中间层
weights_L1=tf.Variable(tf.random_normal([1,10]))  #权重，10个神经元，所以输出为10，位于第一层，所以输入为1
biases_L1=tf.Variable(tf.zeros([1,10]))           #偏见值
wx_plus_b_L1=tf.matmul(x,weights_L1)+biases_L1 #信号总和
L1=tf.nn.tanh(wx_plus_b_L1)
# 定义神经网络输出层
weights_L2=tf.Variable(tf.random_normal([10,1]))
biases_L2=tf.Variable(tf.zeros([1,1]))
wx_plus_b_L2=tf.matmul(L1,weights_L2)+biases_L2  #特别注意，这里的矩阵乘法只能是（输入，权重）不能倒过来，否则输出矩阵形状会出错。
prediction_value=tf.nn.tanh(wx_plus_b_L2)
# 定义二阶代价函数
loss=tf.reduce_mean(tf.square(y-prediction_value))
train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(2000):
        sess.run(train_step,feed_dict={x:xdata,y:ydata})
    p=sess.run(prediction_value,feed_dict={x:xdata})

# 画图
plt.figure()
plt.scatter(xdata,ydata)
plt.plot(xdata,p,'r-',lw=5)
plt.show()
