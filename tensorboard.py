# -*- coding:utf-8 -*- 
'''
    @Author: Tomas Wu 
    @Date: 2019-03-26 23:53:19 
    @Desc: 提高识别率到0.98
'''
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

#参数概要
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean',mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar('stddev',stddev)
        tf.summary.scalar('max',tf.reduce_max(var))
        tf.summary.scalar('min',tf.reduce_min(var))
        tf.summary.histogram('histogram',var)


mnist=input_data.read_data_sets('MNIST_DATA',one_hot=True)

#命名空间
with tf.name_scope('input'):
    batch_size = 100
    n_batch = mnist.train.num_examples // batch_size
    x = tf.placeholder(tf.float32,[None,784],name='x_input')
    y = tf.placeholder(tf.float32,[None,10],name='y_input')
    lr= tf.Variable(0.001,tf.float32,name='learn_rate')

# w = tf.Variable(tf.zeros([784,10])) #但初始化一般不全为0，而是采用截断的正态分布
w1 = tf.Variable(tf.truncated_normal([784,300],stddev=0.1))
b1 = tf.Variable(tf.zeros([300])+0.1)
L1 = tf.nn.tanh(tf.matmul(x,w1)+b1)
variable_summaries(w1)
variable_summaries(b1)

w2 = tf.Variable(tf.truncated_normal([300,100],stddev=0.1))
b2 = tf.Variable(tf.zeros([100])+0.1)
L2 = tf.nn.tanh(tf.matmul(L1,w2)+b2)
variable_summaries(w2)
variable_summaries(b2)



w3 = tf.Variable(tf.truncated_normal([100,10],stddev=0.1))
b3 = tf.Variable(tf.zeros([10])+0.1)
# L3 = tf.nn.tanh(tf.matmul(L2,w3)+b3)
variable_summaries(w3)
variable_summaries(b3)

# w4 = tf.Variable(tf.truncated_normal([50,10],stddev=0.1))
# b4 = tf.Variable(tf.zeros([10])+0.1)
# L4 = tf.nn.tanh(tf.matmul(L3,w4)+b4)
prediction = tf.nn.softmax(tf.matmul(L2,w3)+b3)

# loss = tf.reduce_mean(tf.square(y-prediction))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
tf.summary.scalar('loss',loss )

# train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
train_step = tf.train.AdamOptimizer(lr).minimize(loss)

init = tf.global_variables_initializer()

correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
tf.summary.scalar('accuarcy',accuracy)


#合并所有的summary
merged=tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('logs/',sess.graph)
    for epoch in range(51):
        sess.run(tf.assign(lr,0.001*0.95**epoch))
        for batch in range(n_batch):    
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            summary,_ = sess.run([merged,train_step],feed_dict = {x:batch_xs,y:batch_ys})
        writer.add_summary(summary,epoch)
        test_acc = sess.run(accuracy,feed_dict = {x:mnist.test.images,y:mnist.test.labels})
        train_acc = sess.run(accuracy,feed_dict = {x:mnist.train.images,y:mnist.train.labels})
        print('Iter'+str(epoch)+',Testing Accuracy '+str(test_acc)+' Training Accuracy '+ str(train_acc))
        