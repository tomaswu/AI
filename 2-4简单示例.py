# -*- coding:utf-8 -*- 
'''
    @Author: Tomas Wu 
    @Date: 2019-03-25 23:08:31 
    @Desc: 
'''
import tensorflow as tf
import numpy as np
#生成随机样本
xdata=np.random.rand(100)
ydata=xdata*2+0.4
# 构造线性模型
k=tf.Variable(0.1)
b=tf.Variable(5.)
y=xdata*k+b
# 优化模型（k,b）使其接近样本
# 构造二阶代价函数
loss=tf.reduce_mean(tf.square(ydata-y))
# 定义一个梯度下降法进行训练的优化器
optmizer=tf.train.GradientDescentOptimizer(0.2)
# 定义一个最小化代价函数
train=optmizer.minimize(loss)
init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(2001):
        sess.run(train)
        if step%20==0:
            print(sess.run([k,b]))

