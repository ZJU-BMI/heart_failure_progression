import tensorflow as tf
from rnn_cell import ContextualGRUCellFuseHawkes


def __fuse_hawkes_rnn(num_steps, num_hidden, num_feature, x_placeholder, batch_size, base_intensity,
                      mutual_intensity, task_type, time_interval):
    """
    :param num_steps:
    :param num_hidden:
    :param num_feature:
    :param x_placeholder:
    :param batch_size:
    :param base_intensity:
    :param mutual_intensity:
    :param time_interval:
    :param task_type:
    :return:     loss, prediction, x_placeholder, y_placeholder, batch_size, phase_indicator
    其中 phase_indicator>0代表是测试期，<=0代表是训练期
    """
    zero_state = tf.zeros([batch_size, num_hidden])
    weight_initializer = tf.initializers.orthogonal()
    bias_initializer = tf.initializers.zeros()

    rnn_cell = ContextualGRUCellFuseHawkes(num_hidden=num_hidden, input_length=num_feature,
                                           base_intensity=base_intensity, weight_initializer=weight_initializer,
                                           bias_initializer=bias_initializer, mutual_intensity=mutual_intensity)

    output_list = list()
    x_unstack = tf.unstack(x_placeholder, axis=1)
    time_interval = tf.unstack(time_interval, axis=1)
    state = zero_state

    for i in range(num_steps):
        state = rnn_cell(x_unstack[i], state, task_type, time_interval[i])
        output_list.append(state)
    return output_list


def fuse_hawkes_rnn_model(num_steps, num_hidden, num_feature, keep_rate, event_count=11):
    """
    :param num_steps:
    :param num_hidden:
    :param num_feature:
    :param keep_rate:
    :param event_count:
    :return:
    loss, prediction, x_placeholder, y_placeholder, batch_size, phase_indicator
    其中 phase_indicator>0代表是测试期，<=0代表是训练期
    """
    with tf.name_scope('fuse_hawkes_rnn'):
        with tf.name_scope('data_source'):
            batch_size = tf.placeholder(tf.int32, [], name='batch_size')
            x_placeholder = tf.placeholder(tf.float32, [None, num_steps, num_feature], name='x_placeholder')
            y_placeholder = tf.placeholder(tf.float32, [None, 1], name='y_placeholder')
            phase_indicator = tf.placeholder(tf.int32, shape=[], name="phase_indicator")
            task_type = tf.placeholder(tf.int32, shape=[], name="task_type")
            mutual_intensity = tf.placeholder(tf.float32, shape=[event_count, event_count], name="mutual_intensity")
            base_intensity = tf.placeholder(tf.float32, shape=[event_count, 1], name="base_intensity")
            time_interval = tf.placeholder(tf.float32, shape=[None, num_steps], name="time_interval")

        with tf.name_scope('rnn'):
            # 用于判断是训练阶段还是测试阶段，用于判断是否要加Dropout
            x_dropout = tf.cond(phase_indicator > 0,
                                lambda: x_placeholder,
                                lambda: tf.nn.dropout(x_placeholder, keep_rate))

            output_list = __fuse_hawkes_rnn(num_steps, num_hidden, num_feature, x_dropout, batch_size,
                                            base_intensity, mutual_intensity, task_type, time_interval)

    with tf.variable_scope('output_layer'):
        output_weight = tf.get_variable("weight", [num_hidden, 1], initializer=tf.initializers.orthogonal())
        bias = tf.get_variable('bias', [])

    with tf.name_scope('prediction'):
        unnormalized_prediction = tf.matmul(output_list[-1], output_weight) + bias
        loss = tf.losses.sigmoid_cross_entropy(logits=unnormalized_prediction, multi_class_labels=y_placeholder)
        prediction = tf.sigmoid(unnormalized_prediction)

    return loss, prediction, x_placeholder, y_placeholder, batch_size, phase_indicator, task_type, time_interval, \
        mutual_intensity, base_intensity


def unit_test():
    num_steps = 10
    num_hidden = 20
    num_feature = 30
    keep_rate = 0.8
    fuse_hawkes_rnn_model(num_steps, num_hidden, num_feature, keep_rate)


if __name__ == '__main__':
    unit_test()
