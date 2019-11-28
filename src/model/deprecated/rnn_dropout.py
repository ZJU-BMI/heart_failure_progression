# encoding=utf-8-sig
# 本代码用于测试最普通的RNN的性能
# 本模块中使用的Drop策略是 arXiv 1603.05118 Recurrent Dropout without Memory Loss中所提及的策略
# 也就是使用GRU对当前输入所做产生的隐变量g_t做随机丢弃
# deprecated
import tensorflow as tf
from deprecated.rnn_cell import DropContextualGRUCell


def drop_rnn(num_steps, num_hidden, num_feature, keep_rate, x_placeholder, phase_indicator, batch_size):
    """
    :param num_steps: 时序数据的长度
    :param num_hidden:
    :param num_feature:时序数据的维度
    :param keep_rate:
    :param x_placeholder:
    :param phase_indicator:
    :param batch_size:
    :return: output list with TBD format
    loss, prediction, x_placeholder, y_placeholder, batch_size, phase_indicator
    其中 phase_indicator>0代表是测试期，<=0代表是训练期
    """
    zero_state = tf.zeros([batch_size, num_hidden])
    weight_initializer = tf.initializers.orthogonal()
    bias_initializer = tf.initializers.zeros()

    rnn_cell = DropContextualGRUCell(num_hidden, num_feature, keep_rate, phase_indicator, weight_initializer,
                                     bias_initializer)

    output_list = list()
    x_unstack = tf.unstack(x_placeholder, axis=1)
    state = zero_state
    for i in range(num_steps):
        state = rnn_cell.call(x_unstack[i], state)
        output_list.append(state)
    return output_list


def rnn_drop_model(num_steps, num_hidden, num_feature, keep_rate):
    batch_size = tf.placeholder(tf.int32, [], name='batch_size')
    x_placeholder = tf.placeholder(tf.float32, [None, num_steps, num_feature], name='x_placeholder')
    y_placeholder = tf.placeholder(tf.float32, [None, 1], name='y_placeholder')
    phase_indicator = tf.placeholder(tf.int32, shape=[], name="phase_indicator")

    output_list = drop_rnn(num_steps, num_hidden, num_feature, keep_rate, x_placeholder, phase_indicator, batch_size)
    output_list = tf.reduce_mean(tf.convert_to_tensor(output_list), axis=0)

    with tf.variable_scope('output_layer'):
        output_weight = tf.get_variable("weight", [num_hidden, 1], initializer=tf.initializers.orthogonal())
        bias = tf.get_variable('bias', [])

    unnormalized_prediction = tf.matmul(output_list, output_weight) + bias
    loss = tf.losses.sigmoid_cross_entropy(logits=unnormalized_prediction, multi_class_labels=y_placeholder)

    prediction = tf.sigmoid(unnormalized_prediction)

    return loss, prediction, x_placeholder, y_placeholder, batch_size, phase_indicator
