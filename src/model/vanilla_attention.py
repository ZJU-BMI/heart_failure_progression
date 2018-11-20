# encoding=utf-8-sig
# 所谓Vanilla Attention，就是很单纯的对Hidden State进行加权求和
import tensorflow as tf
from rnn_regularization import regularization_rnn
from rnn_dropout import drop_rnn


def vanilla_attention(num_steps, num_hidden, num_feature, keep_rate, cell_type):
    batch_size = tf.placeholder(tf.int32, [], name='batch_size')
    x_placeholder = tf.placeholder(tf.float32, [None, num_steps, num_feature], name='x_placeholder')
    y_placeholder = tf.placeholder(tf.float32, [None, 1], name='y_placeholder')
    phase_indicator = tf.placeholder(tf.int32, shape=[], name="phase_indicator")

    if cell_type == 'rnn_regularization':
        x_placeholder = tf.cond(phase_indicator > 0,
                                lambda: x_placeholder,
                                lambda: tf.nn.dropout(x_placeholder, keep_rate))

        output_list = regularization_rnn(num_steps, num_hidden, num_feature, x_placeholder, phase_indicator, batch_size)
    elif cell_type == 'rnn_dropout':
        output_list = drop_rnn(num_steps, num_hidden, num_feature, keep_rate, x_placeholder, phase_indicator,
                               batch_size)
    else:
        raise ValueError('Cell Type Name Invalid')

    with tf.variable_scope('attention_weight'):
        attention_weight = tf.get_variable("attention", [num_steps, 1, 1], initializer=tf.initializers.orthogonal())

    with tf.name_scope('attention_output'):
        attention_weight = attention_weight/tf.reduce_sum(attention_weight)
        output_list = tf.convert_to_tensor(output_list)
        output_list = attention_weight*output_list
        output = tf.reduce_sum(output_list, axis=0)

    with tf.variable_scope('output_weight'):
        output_weight = tf.get_variable("output_weight", [num_hidden, 1], initializer=tf.initializers.orthogonal())
        bias = tf.get_variable('bias', [])
        unnormalized_prediction = tf.matmul(output, output_weight) + bias
        loss = tf.losses.sigmoid_cross_entropy(logits=unnormalized_prediction, multi_class_labels=y_placeholder)

        prediction = tf.sigmoid(unnormalized_prediction)

    return loss, prediction, x_placeholder, y_placeholder, batch_size, phase_indicator
