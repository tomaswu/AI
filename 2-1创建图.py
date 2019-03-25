# -*- coding:utf-8 -*- 
'''
    @Author: Tomas Wu 
    @Date: 2019-03-24 22:16:11 
    @Desc: 
'''
import tensorflow as tf 
m1=tf.constant([[3,3]])
m2=tf.constant([[2],[3]])
# 此处的result即为一个乘法op
result=tf.matmul(m1,m2)
print(result)
sess=tf.Session()
r=sess.run(result)  #op必须在会话之中才会被真正的执行
print(r)
sess.close()