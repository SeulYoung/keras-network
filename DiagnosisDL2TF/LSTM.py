"""
Created on Sat Dec 16 15:26:21 2017

@author: shanpo
"""

import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.contrib import rnn


tf.disable_v2_behavior()

nSampleSize = 7500  # 总样本数
nSig_dim = 576  # 单个样本维度
nLab_dim = 10  # 类别维度

learning_rate = 1e-3
batch_size = tf.placeholder(tf.int32, [])  # 在训练和测试，用不同的 batch_size
input_size = 24  # 每个时刻的输入维数为 24
timestep_size = 24  # 时序长度为24
hidden_size = 128  # 每个隐含层的节点数
layer_num = 3  # LSTM layer 的层数
class_num = nLab_dim  # 类别维数


def getdata(nSampSize=24000):
    # 读取float型二进制数据
    signal = np.fromfile('DLdata90singal.raw', dtype=np.float32)
    labels = np.fromfile('DLdata90labels.raw', dtype=np.float32)
    # 由于matlab 矩阵写入文件是按照【列】优先, 需要按行读取
    mat_sig = np.reshape(signal, [-1, nSampSize])
    mat_lab = np.reshape(labels, [-1, nSampSize])
    mat_sig = mat_sig.T  # 转换成正常样式 【样本序号，样本维度】
    mat_lab = mat_lab.T
    return mat_sig, mat_lab


def zscore(xx):
    # 样本归一化到【-1，1】，逐条对每个样本进行自归一化处理
    max1 = np.max(xx, axis=1)  # 按行或者每个样本，并求出单个样本的最大值
    max1 = np.reshape(max1, [-1, 1])  # 行向量 ->> 列向量
    min1 = np.min(xx, axis=1)  # 按行或者每个样本，并求出单个样本的最小值
    min1 = np.reshape(min1, [-1, 1])  # 行向量 ->> 列向量
    xx = (xx - min1) / (max1 - min1) * 2 - 1
    return xx


def NextBatch(iLen, n_batchsize):
    # iLen: 样本总数
    # n_batchsize: 批处理大小
    # 返回n_batchsize个随机样本（序号）
    ar = np.arange(iLen)  # 生成0到iLen-1，步长为1的序列
    np.random.shuffle(ar)  # 打乱顺序
    return ar[0:n_batchsize]


xs = tf.placeholder(tf.float32, [None, nSig_dim])
ys = tf.placeholder(tf.float32, [None, class_num])
keep_prob = tf.placeholder(tf.float32)

x_input = tf.reshape(xs, [-1, 24, 24])


# 搭建LSTM 模型
def unit_LSTM():
    # 定义一层 LSTM_cell，只需要说明 hidden_size
    lstm_cell = rnn.BasicLSTMCell(num_units=hidden_size, forget_bias=1.0, state_is_tuple=True)
    # 添加 dropout layer, 一般只设置 output_keep_prob
    lstm_cell = rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
    return lstm_cell


# 调用 MultiRNNCell 来实现多层 LSTM
mLSTM_cell = rnn.MultiRNNCell([unit_LSTM() for icnt in range(layer_num)], state_is_tuple=True)

# 用全零来初始化state
init_state = mLSTM_cell.zero_state(batch_size, dtype=tf.float32)
outputs, state = tf.nn.dynamic_rnn(mLSTM_cell, inputs=x_input,
                                   initial_state=init_state, time_major=False)
h_state = outputs[:, -1, :]  # 或者 h_state = state[-1][1]

# 设置 loss function 和 优化器
W = tf.Variable(tf.truncated_normal([hidden_size, class_num], stddev=0.1), dtype=tf.float32)
bias = tf.Variable(tf.constant(0.1, shape=[class_num]), dtype=tf.float32)
y_pre = tf.nn.softmax(tf.matmul(h_state, W) + bias)
# 损失和评估函数
cross_entropy = tf.reduce_mean(tf.reduce_sum(-ys * tf.log(y_pre), reduction_indices=[1]))
train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(ys, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

mydata = getdata()
iTrainSetSize = np.floor(nSampleSize * 3 / 4).astype(int)  # 训练样本个数
iIndex = np.arange(nSampleSize)  # 按照顺序，然后划分训练样本、测试样本
train_index = iIndex[0:iTrainSetSize]
test_index = iIndex[iTrainSetSize:nSampleSize]

train_data = mydata[0][train_index]  # 训练数据
train_y = mydata[1][train_index]  # 训练标签
test_data = mydata[0][test_index]  # 测试数据
test_y = mydata[1][test_index]  # 测试标签

train_x = zscore(train_data)  # 对训练数据进行归一化
test_x = zscore(test_data)  # 对测试数据进行归一化

init = tf.global_variables_initializer()
# 开始训练。
with tf.Session() as sess:
    sess.run(init)
    for icnt in range(1000):
        _batch_size = 100
        intervals = NextBatch(iTrainSetSize, _batch_size)  # 每次从所有样本中随机取100个样本（序号）
        xx = train_x[intervals]
        yy = train_y[intervals]
        if (icnt + 1) % 100 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={
                xs: xx, ys: yy, keep_prob: 1.0, batch_size: _batch_size})
            print("step: " + "{0:4d}".format(icnt + 1) + ",  train acc:" + "{:.4f}".format(train_accuracy))
        sess.run(train_op, feed_dict={xs: xx, ys: yy, keep_prob: 0.9, batch_size: _batch_size})
    bsize = test_x.shape[0]
    test_acc = sess.run(accuracy, feed_dict={xs: test_x, ys: test_y, keep_prob: 1.0, batch_size: bsize})
    print("test acc:" + "{:.4f}".format(test_acc))
