#!/usr/bin/env python
# -*-coding:utf-8 -*-
##*************************************************************************************************************

##*************************************************************************************************************
## **  文件名称：
## **  功能描述：  tensorboard图，方便查看运行过程
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

# 载入数据集
mnist = input_data.read_data_sets("../examples/tutorials/mnist/data", one_hot=True)

# 每个批次的大小
batch_size = 100
# 计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size

#图的保存目录
graph_path="graphs/"

#参数概要
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)#平均值
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)#标准差
        tf.summary.scalar('max', tf.reduce_max(var))#最大值
        tf.summary.scalar('min', tf.reduce_min(var))#最小值
        tf.summary.histogram('histogram', var)#直方图

# 定义两个placeholder
with tf.name_scope("input"):
    x = tf.placeholder(tf.float32, [None, 784] ,name="x-input")
    y = tf.placeholder(tf.float32, [None, 10],name="y-input")

# 创建一个简单的神经网络
with tf.name_scope("lr"):
    lr=tf.Variable(0.001,dtype=tf.float32)

with tf.name_scope("layer1"):
    with tf.name_scope("wights"):
        w1 = tf.Variable(tf.truncated_normal([784, 500], stddev=0.1),name="w1")
        variable_summaries(w1)
    with tf.name_scope("biases"):
        b1 = tf.Variable(tf.zeros([500]) + 0.1,name="b1")
        variable_summaries(b1)
    with tf.name_scope("wx_plus_b"):
        wx1 = tf.matmul(x, w1) + b1
    with tf.name_scope("tanh"):
        l1 = tf.nn.tanh(wx1)

with tf.name_scope("layer2"):
    W2 = tf.Variable(tf.truncated_normal([500, 50], stddev=0.1))
    b2 = tf.Variable(tf.zeros([50]) + 0.1)
    l2 = tf.nn.tanh(tf.matmul(l1, W2) + b2)

with tf.name_scope("layer3"):
    W = tf.Variable(tf.truncated_normal([50, 10], stddev=0.1))
    b = tf.Variable(tf.zeros([10]) + 0.1)
    prediction = tf.nn.softmax(tf.matmul(l2, W) + b)

# 交叉熵
with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
    tf.summary.scalar('loss', loss)

# 使用梯度下降法
with tf.name_scope("train"):
    train_step = tf.train.AdamOptimizer(lr).minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()

with tf.name_scope("accuracy"):
    # 结果存放在一个布尔型列表中
    with tf.name_scope("correct_prediction"):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))  # argmax返回一维张量中最大的值所在的位置
    # 求准确率
    with tf.name_scope("accuracy"):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy",accuracy)

#合并所有summary
merged=tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter(graph_path, sess.graph)
    for epoch in range(100):
        sess.run(tf.assign(lr,0.001*(0.95**epoch)))
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            summary,_=sess.run([merged,train_step], feed_dict={x: batch_xs, y: batch_ys})

        writer.add_summary(summary,epoch)
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        # print("Iter " + str(epoch) + ",Testing Accuracy " + str(acc))
        print("Iter " + str(epoch) + "," + datetime.datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S") + ",Testing Accuracy " + str(acc))

