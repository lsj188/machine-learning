#!/usr/bin/env python
# -*-coding:utf-8 -*-
##*************************************************************************************************************

##*************************************************************************************************************
## **  文件名称：
## **  功能描述：  使用tensorflow mnist 入门实例
## **
## **
## **  输入：
## **  输出：
## **
## **
## **  创建者：骆仕军
## **  创建日期：2017-10-10
## **  修改日期：
## **  修改日志：

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 使用tensorflow自带的工具加载MNIST手写数字集合
mnist = input_data.read_data_sets('../data/', one_hot=True)

#用占位符定义x（输入值）
x = tf.placeholder(tf.float32, [None, 784])

#权重
W = tf.Variable(tf.zeros([784, 10]))

#偏置
b = tf.Variable(tf.zeros([10]))

#模型（预测值）
y = tf.nn.softmax(tf.matmul(x, W) + b)

#用占位符定义y_(期望值，实际值)
y_ = tf.placeholder("float", [None, 10])

#计算交叉熵：代价函数(损失函数）
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

#训练模型（梯度下降算法）
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

#初使化变量
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

#开始训练模型
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

#测试模型
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

#输出结果
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
