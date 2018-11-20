import tensorflow as tf


def revised_rnn(num_steps, num_hidden, num_feature, x_placeholder, batch_size, base_intensity,
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

    rnn_cell = ContextualGRUCellRevised(num_hidden=num_hidden, input_length=num_feature,
                                        weight_initializer=weight_initializer, bias_initializer=bias_initializer)

    output_list = list()
    x_unstack = tf.unstack(x_placeholder, axis=1)
    time_interval = tf.unstack(time_interval, axis=1)
    state = zero_state

    for i in range(num_steps):
        state = rnn_cell(x_unstack[i], state, base_intensity, mutual_intensity, task_type, time_interval[i])
        output_list.append(state)
    return output_list


def revised_rnn_model(num_steps, num_hidden, num_feature, keep_rate, base_intensity, mutual_intensity):
    """
    :param num_steps:
    :param num_hidden:
    :param num_feature:
    :param keep_rate:
    :param base_intensity:
    :param mutual_intensity:
    :return:
    loss, prediction, x_placeholder, y_placeholder, batch_size, phase_indicator
    其中 phase_indicator>0代表是测试期，<=0代表是训练期
    """
    batch_size = tf.placeholder(tf.int32, [], name='batch_size')
    x_placeholder = tf.placeholder(tf.float32, [None, num_steps, num_feature], name='x_placeholder')
    y_placeholder = tf.placeholder(tf.float32, [None, 1], name='y_placeholder')
    phase_indicator = tf.placeholder(tf.int32, shape=[], name="phase_indicator")
    task_type = tf.placeholder(tf.int32, shape=[], name="task_type")
    mutual_intensity = tf.convert_to_tensor(mutual_intensity, dtype=tf.float32)
    base_intensity = tf.convert_to_tensor(base_intensity, dtype=tf.float32)
    time_interval = tf.placeholder(tf.float32, shape=[None, num_steps], name="time_interval")

    # 用于判断是训练阶段还是测试阶段，用于判断是否要加Dropout
    x_dropout = tf.cond(phase_indicator > 0,
                        lambda: x_placeholder,
                        lambda: tf.nn.dropout(x_placeholder, keep_rate))

    output_list = revised_rnn(num_steps, num_hidden, num_feature, x_dropout, batch_size,
                              base_intensity, mutual_intensity, task_type, time_interval)

    with tf.variable_scope('output_layer'):
        output_weight = tf.get_variable("weight", [num_hidden, 1], initializer=tf.initializers.orthogonal())
        bias = tf.get_variable('bias', [])

    unnormalized_prediction = tf.matmul(output_list[-1], output_weight) + bias
    loss = tf.losses.sigmoid_cross_entropy(logits=unnormalized_prediction, multi_class_labels=y_placeholder)

    prediction = tf.sigmoid(unnormalized_prediction)

    return loss, prediction, x_placeholder, y_placeholder, batch_size, phase_indicator, task_type, time_interval


class ContextualGRUCellRevised(object):
    def __init__(self, num_hidden, input_length, weight_initializer, bias_initializer, event_count=11):
        """
        :param num_hidden:
        :param weight_initializer:
        :param bias_initializer:
        """
        self.__num_hidden = num_hidden
        self.__weight_initializer = weight_initializer
        self.__bias_initializer = bias_initializer
        self.__input_length = input_length
        self.__event_count = event_count

    def __call__(self, input_x, prev_hidden_state, base_intensity, mutual_intensity, task_type, time_interval,
                 event_count=11, omega=0.006):
        """
        :param input_x:
        :param prev_hidden_state:
        :param base_intensity:
        :param mutual_intensity:
        :param task_type:
        :param event_count:
        :param omega:
        :return:
        """
        input_length = self.__input_length
        num_hidden = self.__num_hidden
        weight_initializer = self.__weight_initializer
        bias_initializer = self.__bias_initializer

        event_list = tf.slice(input_x, [0, 0], [-1, event_count])

        # 获取互激发矩阵强度
        # 互激发矩阵中的元素e_ij指代第i类事件激发第j类事件的强度
        mutual_intensity_vector = tf.expand_dims(mutual_intensity[task_type, :], axis=1)
        mutual_intensity_weight = tf.matmul(event_list, mutual_intensity_vector)

        # 基于上述强度指导三个权重的分配
        time_interval = tf.expand_dims(time_interval, axis=1)
        time_intensity = tf.exp(-omega * time_interval)
        hawkes_intensity = time_intensity*mutual_intensity_weight + base_intensity[task_type]

        with tf.variable_scope('DropGRU_Parameter', reuse=tf.AUTO_REUSE):
            weight_z = tf.get_variable('w_z', [num_hidden, 1], initializer=weight_initializer)
            weight_r = tf.get_variable('w_r', [num_hidden, 1], initializer=weight_initializer)
            weight_g = tf.get_variable('w_t', [num_hidden+input_length, 1], initializer=weight_initializer)
            bias_z = tf.get_variable('b_z', [num_hidden, 1], initializer=bias_initializer)
            bias_r = tf.get_variable('b_r', [num_hidden, 1], initializer=bias_initializer)
            bias_g = tf.get_variable('b_t', [num_hidden, 1], initializer=bias_initializer)

        with tf.name_scope('DropGRU_Internal'):
            z_t = tf.sigmoid(tf.matmul(weight_z, tf.transpose(hawkes_intensity)) + bias_z)
            r_t = tf.sigmoid(tf.matmul(weight_r, tf.transpose(hawkes_intensity)) + bias_r)

            concat_2 = tf.transpose(tf.concat([input_x, tf.transpose(r_t)*prev_hidden_state], axis=1))
            g_t = tf.matmul(tf.transpose(weight_g), concat_2) + bias_g
            new_state = tf.transpose((1 - z_t) * tf.transpose(prev_hidden_state) + z_t * g_t)
        return new_state
