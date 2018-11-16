# encoding=utf-8-sig
# 本代码用于测试最普通的RNN的性能
# 本模块中使用的Drop策略是 arXiv 1409.2329 recurrent neural regularization中所提及的策略
# 也就是Dropout只在输入处进行输入信号的随机丢弃
import tensorflow as tf
from rnn_cell import ContextualGRUCell


def regularization_rnn(num_steps, num_hidden, num_feature, x_placeholder, phase_indicator, batch_size):
    """
    :param num_steps: 时序数据的长度
    :param num_hidden:
    :param num_feature:时序数据的维度
    :param x_placeholder:
    :param phase_indicator:
    :param batch_size:
    :return:
    loss, prediction, x_placeholder, y_placeholder, batch_size, phase_indicator
    其中 phase_indicator>0代表是测试期，<=0代表是训练期
    """
    zero_state = tf.zeros([batch_size, num_hidden])
    weight_initializer = tf.initializers.orthogonal()
    bias_initializer = tf.initializers.zeros()

    rnn_cell = ContextualGRUCell(num_hidden, num_feature, phase_indicator, weight_initializer, bias_initializer)

    output_list = list()
    x_unstack = tf.unstack(x_placeholder, axis=1)
    state = zero_state
    for i in range(num_steps):
        state = rnn_cell.call(x_unstack[i], state)
        output_list.append(state)
    return output_list


def regularization_rnn_model(num_steps, num_hidden, num_feature, keep_rate):
    """
    :param num_steps:
    :param num_hidden:
    :param num_feature:
    :param keep_rate:
    :return:
    loss, prediction, x_placeholder, y_placeholder, batch_size, phase_indicator
    其中 phase_indicator>0代表是测试期，<=0代表是训练期
    """
    batch_size = tf.placeholder(tf.int32, [], name='batch_size')
    x_placeholder = tf.placeholder(tf.float32, [None, num_steps, num_feature], name='x_placeholder')
    y_placeholder = tf.placeholder(tf.float32, [None, 1], name='y_placeholder')
    phase_indicator = tf.placeholder(tf.int32, shape=[], name="phase_indicator")

    # 用于判断是训练阶段还是测试阶段，用于判断是否要加Dropout
    x_dropout = tf.cond(phase_indicator > 0,
                        lambda: x_placeholder,
                        lambda: tf.nn.dropout(x_placeholder, keep_rate))

    output_list = regularization_rnn(num_steps, num_hidden, num_feature, x_dropout, phase_indicator, batch_size)

    with tf.variable_scope('output_layer'):
        output_weight = tf.get_variable("weight", [num_hidden, 1], initializer=tf.initializers.orthogonal())
        bias = tf.get_variable('bias', [])

    unnormalized_prediction = tf.matmul(output_list[-1], output_weight) + bias
    loss = tf.losses.sigmoid_cross_entropy(logits=unnormalized_prediction, multi_class_labels=y_placeholder)

    prediction = tf.sigmoid(unnormalized_prediction)

    return loss, prediction, x_placeholder, y_placeholder, batch_size, phase_indicator

