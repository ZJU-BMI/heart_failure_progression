# encoding=utf-8-sig
# 使用rnn_dropout作为RNN核
# 使用arXiv 1703.03130 A STRUCTURED SELF-ATTENTIVE SENTENCE EMBEDDING 实现Attention
import tensorflow as tf
from rnn_dropout import drop_rnn


def self_attention(num_steps, num_hidden, num_feature, keep_rate, s1=2, s2=2):
    batch_size = tf.placeholder(tf.int32, [], name='batch_size')
    x_placeholder = tf.placeholder(tf.float32, [None, num_steps, num_feature], name='x_placeholder')
    y_placeholder = tf.placeholder(tf.float32, [None, 1], name='y_placeholder')
    phase_indicator = tf.placeholder(tf.int32, shape=[], name="phase_indicator")

    output_list = drop_rnn(num_steps, num_hidden, num_feature, keep_rate, x_placeholder, phase_indicator, batch_size)
    with tf.variable_scope('attention_weight'):
        weight_s1 = tf.get_variable("weight_s1", [s1, num_hidden], initializer=tf.initializers.orthogonal())
        weight_s2 = tf.get_variable("weight_s2", [s2, s1], initializer=tf.initializers.orthogonal())

    with tf.name_scope('attention_output'):
        # TBD -> BTD
        output_list = tf.transpose(tf.convert_to_tensor(output_list), [1, 0, 2])

        # 二维与三维的矩阵相乘，又碰到这种Batch_Size未定义的情况，使用Map_FN映射完成
        # self attention 算法
        # 参考 https://github.com/flrngel/Self-Attentive-tensorflow 中的实现
        attention = tf.tanh(tf.map_fn(lambda x: tf.matmul(weight_s1, tf.transpose(x)), output_list))
        attention = tf.nn.softmax(tf.map_fn(lambda x: tf.matmul(weight_s2, x), attention))
        attention_t = tf.transpose(attention, perm=[0, 2, 1])
        penalty = tf.reduce_mean(tf.square(tf.norm(tf.matmul(attention, attention_t) - 0.5, axis=[-2, -1], ord='fro')))

        attention_flatten = tf.reshape(tf.matmul(attention, output_list), [-1, s2*num_hidden])

    with tf.variable_scope('output_weight'):
        output_weight = tf.get_variable("output_weight", [s2*num_hidden, 1], initializer=tf.initializers.orthogonal())
        bias = tf.get_variable('bias', [])
        unnormalized_prediction = tf.matmul(attention_flatten, output_weight) + bias
        loss = tf.losses.sigmoid_cross_entropy(logits=unnormalized_prediction,
                                               multi_class_labels=y_placeholder) + penalty

    prediction = tf.sigmoid(unnormalized_prediction)

    return loss, prediction, x_placeholder, y_placeholder, batch_size, phase_indicator
