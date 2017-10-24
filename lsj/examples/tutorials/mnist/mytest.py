#!/usr/bin/env python
# -*-coding:utf-8 -*-
##*************************************************************************************************************

##*************************************************************************************************************
## **  文件名称：  test_pyspark_log.py
## **  功能描述：  机器学习，识别图片数字（0-9）
## **
## **
## **  输入：
## **  输出：
## **
## **
## **  创建者：骆仕军
## **  创建日期：2017-07-26
## **  修改日期：
## **  修改日志：
## **
## **
## **
## **
## **
## **
## ** ---------------------------------------------------------------------------------------
## **
## ** ---------------------------------------------------------------------------------------
## **
## ** 程序调用格式：test_pyspark_log.py $version
## ** eg:pyspark test_pyspark_log.py v2.0
## **
## ******************************************************************************************
## ** 重庆金融资产交易所
## ** All Rights Reserved.
## ******************************************************************************************
## **
## **参数说明：
## **    1、
## **    2、
## **    3、

#引用包
import tensorflow as tf
import time

#下载用于训练和测试的mnist数据集的源码
import lsj.examples.tutorials.mnist.input_data as input_data  # 调用input_data
mnist = input_data.read_data_sets('data/', one_hot=True)

# 创建InteractiveSession，如果你没有使用InteractiveSession，那么你需要在启动session之前构建整个计算图，然后启动该计算图。
sess=tf.InteractiveSession()

#FEED，定义两个操作占位符
#x：是一个2维的浮点数张量，其中784是一张展平的MNIST图片的维度，None：表示值不定，这里代表第一个维度
# y_：是一个2维张量，其中每一行为一个10维的one-hot向量,用于代表对应某一MNIST图片的类别
# 只是占未，并无特定值
x=tf.placeholder("float",shape=[None,784])
y_=tf.placeholder("float",shape=[None,10])

#变量
# 变量需要通过seesion初始化后，才能在session中使用
# W：权重，b：偏置量
#在这个例子里，我们把W和b都初始化为零向量。W是一个784x10的矩阵（因为我们有784个特征和10个输出值）。b是一个10维的向量（因为我们有10个分类）。
W=tf.Variable(tf.zeros([784,10]))   #初始为0
b=tf.Variable(tf.zeros([10]))       #初始为0

#初化全部变量
sess.run(tf.initialize_all_variables())

# 类别预测与损失函数
# 实现我们的回归模型，把向量化后的图片x和权重矩阵W相乘，加上偏置b，然后计算每个分类的softmax概率值。
y=tf.nn.softmax(tf.matmul(x,W)+b)

# 为训练过程指定最小化误差用的损失函数，损失函数是目标类别和预测类别之间的交叉熵。
#tf.reduce_sum把minibatch里的每张图片的交叉熵值都加起来了。计算的交叉熵是指整个minibatch的。
cross_entropy=tf.reduce_sum(y_*tf.log(y))

#训练模型
# 已经定义好模型和训练用的损失函数，那么用TensorFlow进行训练就很简单了。因为TensorFlow知道整个计算图，它可以使用自动微分法找到对于各个变量的损失的梯度值。
# 用最速下降法让交叉熵下降，步长为0.01.
train_step =tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 这一行代码实际上是用来往计算图上添加一个新操作，其中包括计算梯度，计算每个参数的步长变化，并且计算出新的参数值。
# 返回的train_step操作对象，在运行时会使用梯度下降来更新参数。因此，整个模型的训练可以通过反复地运行train_step来完成。
for i in range(1000):
    batch=mnist.train.next_batch(50)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})

# 模型检验
# tf.argmax(y,1)返回的是模型对于任一输入x预测到的标签值，而 tf.argmax(y_,1) 代表正确的标签，
# 我们可以用 tf.equal 来检测我们的预测是否真实标签匹配(索引位置一样表示匹配)。
correct_prediction =tf.equal(tf.argmax(y,1),tf.argmax(y_,1))

#为了计算我们分类的准确率，我们将布尔值转换为浮点数来代表对、错，然后取平均值。例如：[True, False, True, True]变为[1,0,1,1]，计算出平均值为0.75。
accuracy=tf.reduce_mean(tf.cast(correct_prediction,"float"))

# 最后，我们可以计算出在测试数据上的准确率，大概是91%。
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

#============
#构建一个多层卷积网络
# 初始化
# 为了创建这个模型，我们需要创建大量的权重和偏置项。这个模型中的权重在初始化时应该加入少量的噪声来打破对称性以及避免0梯度。
# 由于我们使用的是ReLU神经元，因此比较好的做法是用一个较小的正数来初始化偏置项，以避免神经元节点输出恒为0的问题（dead neurons）。
# 为了不在建立模型的时候反复做初始化操作，定义两个函数用于初始化。
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

#卷积和池化
#TensorFlow在卷积和池化上有很强的灵活性。我们怎么处理边界？步长应该设多大？在这个实例里，我们会一直使用vanilla版本。
# 我们的卷积使用1步长（stride size），0边距（padding size）的模板，保证输出和输入是同一个大小。
# 我们的池化用简单传统的2x2大小的模板做max pooling。为了代码更简洁，我们把这部分抽象成一个函数。
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# 第一层卷积
# 现在我们可以开始实现第一层了。它由一个卷积接一个max pooling完成。卷积在每个5x5的patch中算出32个特征。
# 卷积的权重张量形状是[5, 5, 1, 32]，前两个维度是patch的大小，接着是输入的通道数目，最后是输出的通道数目。 而对于每一个输出通道都有一个对应的偏置量。
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

#为了用这一层，我们把x变成一个4d向量，其第2、第3维对应图片的宽、高，最后一维代表图片的颜色通道数(因为是灰度图所以这里的通道数为1，如果是rgb彩色图，则为3)。
x_image = tf.reshape(x, [-1,28,28,1])

#We then convolve x_image with the weight tensor, add the bias, apply the ReLU function, and finally max pool.
#我们把x_image和权值向量进行卷积，加上偏置项，然后应用ReLU激活函数，最后进行max pooling。
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#第二层卷积
#为了构建一个更深的网络，我们会把几个类似的层堆叠起来。第二层中，每个5x5的patch会得到64个特征。
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#密集连接层
#现在，图片尺寸减小到7x7，我们加入一个有1024个神经元的全连接层，用于处理整个图片。我们把池化层输出的张量reshape成一些向量，乘上权重矩阵，加上偏置，然后对其使用ReLU。
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#Dropout为了减少过拟合，我们在输出层之前加入dropout。我们用一个placeholder来代表一个神经元的输出在dropout中保持不变的概率。
# 这样我们可以在训练过程中启用dropout，在测试过程中关闭dropout。 TensorFlow的tf.nn.dropout操作除了可以屏蔽神经元的输出外，还会自动处理神经元输出值的scale。
# 所以用dropout的时候可以不用考虑scale。
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#输出层
#最后，我们添加一个softmax层，就像前面的单层softmax regression一样。
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

#训练和评估模型
#这个模型的效果如何呢？
#为了进行训练和评估，我们使用与之前简单的单层SoftMax神经网络模型几乎相同的一套代码，只是我们会用更加复杂的ADAM优化器来做梯度最速下降，在feed_dict中加入额外的参数keep_prob来控制dropout比例。然后每100次迭代输出一次日志。
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.initialize_all_variables())
# 在Tensorboard里可以看到图的结构
writer = tf.summary.FileWriter('G:\logs\logistic_reg', sess.graph)
start_time = time.time()
for i in range(20000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print ("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print('Total time: {0} seconds'.format(time.time() - start_time))
print('Optimization Finished!')

print ("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

writer.close()