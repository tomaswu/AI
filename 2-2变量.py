# -*- coding:utf-8 -*- 
'''
    @Author: Tomas Wu 
    @Date: 2019-03-24 22:21:34 
    @Desc: 
'''

import tensorflow as tf

x=tf.Variable([1,2])
a=tf.constant([3,3])

sub=tf.subtract(x,a)
add=tf.add(x,sub)

init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(sub))
    print(sess.run(add))