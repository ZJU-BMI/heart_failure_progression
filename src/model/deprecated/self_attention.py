# encoding=utf-8-sig
# 使用rnn_dropout作为RNN核
# 使用arXiv 1703.03130 A STRUCTURED SELF-ATTENTIVE SENTENCE EMBEDDING 实现Attention
# 具体实现参考 https://github.com/flrngel/Self-Attentive-tensorflow
import tensorflow as tf
from rnn_regularization import regularization_rnn
from deprecated.rnn_dropout import drop_rnn


def self_attention(num_steps, num_hidden, num_feature, keep_rate, cell_type, s1=2, s2=2):
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
        weight_s1 = tf.get_variable("weight_s1", [s1, num_hidden], initializer=tf.initializers.orthogonal())
        weight_s2 = tf.get_variable("weight_s2", [s2, s1], initializer=tf.initializers.orthogonal())

    with tf.name_scope('attention_output'):
        # TBD -> BDT
        output_list = tf.transpose(tf.convert_to_tensor(output_list), [1, 2, 0])

        # 二维与三维的矩阵相乘，又碰到这种Batch_Size未定义的情况，使用Map_FN映射完成
        attention = tf.tanh(tf.map_fn(lambda x: tf.matmul(weight_s1, x), output_list))
        attention = tf.nn.softmax(tf.map_fn(lambda x: tf.matmul(weight_s2, x), attention))

        penalty = tf.map_fn(lambda x: tf.matmul(x, tf.transpose(x)), attention) - 1
        penalty = tf.reduce_sum(tf.square(penalty))

        # 此处的三维矩阵不能直接相乘，需要通过组合相乘的办法解决问题
        concat = tf.concat([attention, output_list], axis=1)
        m_matrix = tf.map_fn(
            lambda x: tf.matmul(
                tf.slice(x, [0, 0], [s2, num_steps]),
                tf.transpose(tf.slice(x, [s2, 0], [num_hidden, num_steps]))
            ), concat)
        attention_flatten = tf.reshape(m_matrix, [-1, s2*num_hidden])

    with tf.variable_scope('output_weight'):
        output_weight = tf.get_variable("output_weight", [s2*num_hidden, 1], initializer=tf.initializers.orthogonal())
        bias = tf.get_variable('bias', [])

    with tf.name_scope('attention_output'):
        unnormalized_prediction = tf.matmul(attention_flatten, output_weight) + bias
        loss = tf.losses.sigmoid_cross_entropy(logits=unnormalized_prediction,
                                               multi_class_labels=y_placeholder)
        # + penalty
        prediction = tf.sigmoid(unnormalized_prediction)

    return loss, prediction, x_placeholder, y_placeholder, batch_size, phase_indicator
