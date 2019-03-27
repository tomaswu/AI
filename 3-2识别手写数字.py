# -*- coding:utf-8 -*- 
'''
    @Author: Tomas Wu 
    @Date: 2019-03-26 23:53:19 
    @Desc: 第三周和第四周的课程程序均在此处
'''
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets('MNIST_DATA',one_hot=True)

batch_size = 50
n_batch = mnist.train.num_examples // batch_size

x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])
keep_prob = tf.placeholder(tf.float32)

# w = tf.Variable(tf.zeros([784,10])) #但初始化一般不全为0，而是采用截断的正态分布
w1 = tf.Variable(tf.truncated_normal([784,200],stddev=0.1))
b1 = tf.Variable(tf.zeros([200])+0.1)
L1 = tf.nn.tanh(tf.matmul(x,w1)+b1)
L1_drop=tf.nn.dropout(L1,keep_prob)

w2 = tf.Variable(tf.truncated_normal([200,100],stddev=0.1))
b2 = tf.Variable(tf.zeros([100])+0.1)
L2 = tf.nn.tanh(tf.matmul(L1_drop,w2)+b2)
L2_drop=tf.nn.dropout(L2,keep_prob)

w3 = tf.Variable(tf.truncated_normal([100,50],stddev=0.1))
b3 = tf.Variable(tf.zeros([50])+0.1)
L3 = tf.nn.tanh(tf.matmul(L2_drop,w3)+b3)
L3_drop=tf.nn.dropout(L3,keep_prob)

w4 = tf.Variable(tf.truncated_normal([50,10],stddev=0.1))
b4 = tf.Variable(tf.zeros([10])+0.1)
L4 = tf.nn.tanh(tf.matmul(L3_drop,w4)+b4)
prediction = tf.nn.softmax(tf.matmul(L3_drop,w4)+b4)

# loss = tf.reduce_mean(tf.square(y-prediction))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
# train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

init = tf.global_variables_initializer()

correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(21):
        for batch in range(n_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict = {x:batch_xs,y:batch_ys,keep_prob:1.0})
        test_acc = sess.run(accuracy,feed_dict = {x:mnist.test.images,y:mnist.test.labels,keep_prob:1.0})
        train_acc = sess.run(accuracy,feed_dict = {x:mnist.train.images,y:mnist.train.labels,keep_prob:1.0})
        print('Iter'+str(epoch)+',Testing Accuracy '+str(test_acc)+' Training Accuracy '+ str(train_acc))