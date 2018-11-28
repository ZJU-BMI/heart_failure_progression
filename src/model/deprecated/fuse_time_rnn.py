import tensorflow as tf
from deprecated.rnn_cell import ContextualGRUCellFuseTimeDecay


def __fuse_time_rnn(num_steps, num_hidden, num_feature, x_placeholder, batch_size, time_interval):
    """
    :param num_steps:
    :param num_hidden:
    :param num_feature:
    :param x_placeholder:
    :param batch_size:
    :param time_interval:
    :return:     loss, prediction, x_placeholder, y_placeholder, batch_size, phase_indicator
    其中 phase_indicator>0代表是测试期，<=0代表是训练期
    """
    zero_state = tf.zeros([batch_size, num_hidden])
    weight_initializer = tf.initializers.orthogonal()
    bias_initializer = tf.initializers.zeros()

    rnn_cell = ContextualGRUCellFuseTimeDecay(num_hidden=num_hidden, input_length=num_feature,
                                              weight_initializer=weight_initializer, bias_initializer=bias_initializer)

    output_list = list()
    x_unstack = tf.unstack(x_placeholder, axis=1)
    time_interval = tf.unstack(time_interval, axis=1)
    state = zero_state

    for i in range(num_steps):
        state = rnn_cell(x_unstack[i], state, time_interval[i])
        output_list.append(state)
    return output_list


def fuse_time_rnn_model(num_steps, num_hidden, num_feature, keep_rate):
    """
    :param num_steps:
    :param num_hidden:
    :param num_feature:
    :param keep_rate:
    :return:
    loss, prediction, x_placeholder, y_placeholder, batch_size, phase_indicator
    其中 phase_indicator>0代表是测试期，<=0代表是训练期
    """
    with tf.name_scope('fuse_time_rnn'):
        with tf.name_scope('data_source'):
            batch_size = tf.placeholder(tf.int32, [], name='batch_size')
            x_placeholder = tf.placeholder(tf.float32, [None, num_steps, num_feature], name='x_placeholder')
            y_placeholder = tf.placeholder(tf.float32, [None, 1], name='y_placeholder')
            phase_indicator = tf.placeholder(tf.int32, shape=[], name="phase_indicator")
            time_interval = tf.placeholder(tf.float32, shape=[None, num_steps], name="time_interval")

            x_dropout = tf.cond(phase_indicator > 0,
                                lambda: x_placeholder,
                                lambda: tf.nn.dropout(x_placeholder, keep_rate))

            output_list = __fuse_time_rnn(num_steps, num_hidden, num_feature, x_dropout, batch_size, time_interval)

    with tf.variable_scope('output_layer'):
        output_weight = tf.get_variable("weight", [num_hidden, 1], initializer=tf.initializers.orthogonal())
        bias = tf.get_variable('bias', [])

    with tf.name_scope('prediction'):
        unnormalized_prediction = tf.matmul(output_list[-1], output_weight) + bias
        loss = tf.losses.sigmoid_cross_entropy(logits=unnormalized_prediction, multi_class_labels=y_placeholder)
        prediction = tf.sigmoid(unnormalized_prediction)

    return loss, prediction, x_placeholder, y_placeholder, batch_size, phase_indicator, time_interval


def unit_test():
    num_steps = 10
    num_hidden = 20
    num_feature = 30
    keep_rate = 0.8
    fuse_time_rnn_model(num_steps, num_hidden, num_feature, keep_rate)


if __name__ == '__main__':
    unit_test()
