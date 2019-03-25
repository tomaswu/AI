# -*- coding:utf-8 -*- 
'''
    @Author: Tomas Wu 
    @Date: 2019-03-24 22:28:13 
    @Desc: 
'''
import tensorflow as tf 

input1=tf.constant(3.0)
input2=tf.constant(2.0)
input3=tf.constant(5.0)

add=tf.add(input2,input3)
mul=tf.multiply(input1,add)
# fetch即为可以同时run多个op
with tf.Session() as sess:
    r=sess.run([mul,add])
    print(r)
# feed即为可以定义占位符，在最后run的时候再使用字典将数据传入
input4=tf.placeholder(tf.float32)
input5=tf.placeholder(tf.float32)
output=tf.multiply(input4,input5)
with tf.Session() as sess:
    print(sess.run(output,feed_dict={input4:3.,input5:4.}))