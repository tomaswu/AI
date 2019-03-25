# -*- coding:utf-8 -*- 
'''
    @Author: Tomas Wu 
    @Date: 2019-03-24 22:24:40 
    @Desc: 
'''

import tensorflow as tf

state=tf.Variable(0,name='counter')
new_value=tf.add(state,1)
update=tf.assign(state,new_value)

init=tf.global_variables_initializer() #变量一定要进行初始化才能够在执行过程中使用

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(state))
    for _ in range(5):
        sess.run(update)
        print(sess.run (state))