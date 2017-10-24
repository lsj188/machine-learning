#!/usr/bin/env python
# -*-coding:utf-8 -*-
##*************************************************************************************************************

##*************************************************************************************************************
## **  文件名称：
## **  功能描述：  CNN卷积神经网络
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

# 初使化权值
def init_weight(name,shape):
    w=tf.truncated_normal(shape,stddev=0.1)  #生成一个截断的正态分布标准差为0.1
    return tf.Variable(w,name=name)

# 初使化偏置
def init_baise(name,shape):
    baise=tf.constant(0.1,shape=shape)
    return tf.Variable(baise,name=name)

# 卷积层
def conv2d(x,w):
    # x input tensor of shape `[batch, in_height, in_width, in_channels]`
    # W filter / kernel tensor of shape [filter_height, filter_width, in_channels, out_channels]
    # `strides[0] = strides[3] = 1`. strides[1]代表x方向的步长，strides[2]代表y方向的步长
    # padding: A `string` from: `"SAME", "VALID"`
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding="SAME")

# 池化层
def max_pool_2x2(x):
    #ksize shape=[1,x,y,1]
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

# 命名空间
with tf.name_scope("input"):
    # 定义x,y placeholder
    x=tf.placeholder(tf.float32,[None,784],name="x-input")
    y=tf.placeholder(tf.float32,[None,10],name="y-input")
    with tf.name_scope("x_image"):
        # 将x转化为[batch,in_height,in_width,in_channels]
        x_image=tf.reshape(x,shape=[-1,28,28,1],name="x_image")


# 第一层卷积网络
with tf.name_scope("conv1"):
    # 初始化第一个卷积层的权值和偏置
    with tf.name_scope("w_conv1"):
        w_conv1=init_weight("w_conv1",[5,5,1,32]) #5*5卷积核（过滤器或采样窗口），32个卷积核从1个平面抽取特征
    with tf.name_scope("b_conv1"):
        b_conv1=init_baise("b_conv1",[32])  # 每个卷积核的偏置

    # 把x_image和权值向量进行卷积，再加上偏置值，然后应用于relu激活函数
    with tf.name_scope("conv2d_1"):
        conv2d_1=conv2d(x_image,w_conv1)+b_conv1
    with tf.name_scope("relu_1"):
        relu_1=tf.nn.relu(conv2d_1)
    with tf.name_scope("h_pool_1"):
        h_pool_1=max_pool_2x2(relu_1)

# 第二层卷积
with tf.name_scope("conv2"):
    with tf.name_scope("w_conv2"):
        w_conv2=init_weight("w_conv2",[5,5,32,64])
    with tf.name_scope("b_conv2"):
        b_conv2=init_baise("b_conv2",[64])

    with tf.name_scope("conv2d_2"):
        conv2d_2=conv2d(h_pool_1,w_conv2)+b_conv2
    with tf.name_scope("relu_2"):
        relu_2=tf.nn.relu(conv2d_2)
    with tf.name_scope("h_pool_2"):
        h_pool_2=max_pool_2x2(relu_2)

# 输入：28*28*1
# 经过第一层卷积变为28*28*32，经过第一层池化后变为14*14*32
# 经过第二层卷积变为14*14*64，经过第二层池化后变为7*7*64
# 全连接层
with tf.name_scope("fc1"):
    with tf.name_scope("fc1_w"):
        fc1_w=init_weight("fc1_w",[7*7*64,1024])
    with tf.name_scope("fc1_b"):
        fc1_b=init_baise("fc1_b",[1024])

    #  将池化层2输出转化为一维
    with tf.name_scope("h_pool_2_t_1"):
        h_pool_2_t_1=tf.reshape(h_pool_2,[-1,7*7*64],name="h_pool_2_t_1")
    with tf.name_scope("fc1_wx_plus_b"):
        fc1_wx_plus_b=tf.matmul(h_pool_2_t_1,fc1_w)+fc1_b
    with tf.name_scope("fc1_relu"):
        fc1_relu=tf.nn.relu(fc1_wx_plus_b)

    #keep_prob用来表示神经元的输出概率
    with tf.name_scope('keep_prob'):
        keep_prob = tf.placeholder(tf.float32,name='keep_prob')
    with tf.name_scope('h_fc1_drop'):
        h_fc1_drop = tf.nn.dropout(fc1_relu,keep_prob,name='h_fc1_drop')

with tf.name_scope("fc2"):
    with tf.name_scope("fc2_w"):
        fc2_w=init_weight("fc2_w",[1024,10])
    with tf.name_scope("fc2_b"):
        fc2_b=init_baise("fc2_b",[10])
    with tf.name_scope("fc2_wx_plus_b"):
        fc2_wx_plus_b=tf.matmul(h_fc1_drop,fc2_w)+fc2_b
    with tf.name_scope("softmax"):
        prediction=tf.nn.softmax(fc2_wx_plus_b)

with tf.name_scope("cross_entropy"):
    cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction),name="cross_entropy")
    tf.summary.scalar("cross_entropy",cross_entropy)

with tf.name_scope("train"):
    train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# 求准确率
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        # 结果存放在一个布尔列表中
        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))  # argmax返回一维张量中最大的值所在的位置
    with tf.name_scope('accuracy'):
        # 求准确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

# 合并所有的summary
merged = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter(graph_path+'/train', sess.graph)
    test_writer = tf.summary.FileWriter(graph_path+'/test', sess.graph)
    for i in range(1001):
        # 训练模型
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.7})
        # 记录训练集计算的参数
        summary = sess.run(merged, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})
        train_writer.add_summary(summary, i)
        # 记录测试集计算的参数
        batch_xs, batch_ys = mnist.test.next_batch(batch_size)
        summary = sess.run(merged, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})
        test_writer.add_summary(summary, i)

        if i % 100 == 0:
            test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
            train_acc = sess.run(accuracy, feed_dict={x: mnist.train.images[:10000], y: mnist.train.labels[:10000],
                                                      keep_prob: 1.0})
            print("Iter " + str(i) + ", Testing Accuracy= " + str(test_acc) + ", Training Accuracy= " + str(train_acc))


