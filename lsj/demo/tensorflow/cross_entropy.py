#!/usr/bin/env python
# -*-coding:utf-8 -*-
##*************************************************************************************************************

##*************************************************************************************************************
## **  文件名称：
## **  功能描述：  交叉熵用作代码函数
## **
## **
## **  输入：
## **  输出：
## **
## **
## **  创建者：骆仕军
## **  创建日期：2017-10-09
## **  修改日期：
## **  修改日志：
import datetime
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#载入数据集
mnist = input_data.read_data_sets("../../../test_data/img_data", one_hot=True)

#每个批次的大小
batch_size = 100
#计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size

#定义两个placeholder
x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])

#创建一个简单的神经网络
# W = tf.Variable(tf.zeros([784,10]))
# b = tf.Variable(tf.zeros([10]))
W1 = tf.Variable(tf.truncated_normal([784,500],stddev=0.1))
b1 = tf.Variable(tf.zeros([500])+0.1)
l1= tf.nn.tanh(tf.matmul(x,W1)+b1)

W = tf.Variable(tf.truncated_normal([500,10],stddev=0.1))
b = tf.Variable(tf.zeros([10])+0.1)
prediction = tf.nn.softmax(tf.matmul(l1,W)+b)

#二次代价函数
# loss = tf.reduce_mean(tf.square(y-prediction))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
#使用梯度下降法
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

#初始化变量
init = tf.global_variables_initializer()

#结果存放在一个布尔型列表中
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))#argmax返回一维张量中最大的值所在的位置
#求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(100):
        for batch in range(n_batch):
            batch_xs,batch_ys =  mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})
        
        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        # print("Iter " + str(epoch) + ",Testing Accuracy " + str(acc))
        print("Iter " + str(epoch) + "," + datetime.datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S") + ",Testing Accuracy " + str(acc))

