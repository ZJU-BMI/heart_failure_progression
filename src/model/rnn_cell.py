import tensorflow as tf
# 此处虽然定义了ContextualGRUCell的定义
# 但是Contextual GRU其实原版GRU在算法上没有区别，其Contextual更像是在语义上进行定义


class ContextualGRUCell(object):
    def __init__(self, num_hidden, input_length, phase_indicator, weight_initializer, bias_initializer):
        """
        :param num_hidden:
        :param weight_initializer:
        :param bias_initializer:
        :param phase_indicator: phase_indicator>0代表是测试期，phase_indicator<=0代表是训练期
        """
        self.__num_hidden = num_hidden
        self.__weight_initializer = weight_initializer
        self.__bias_initializer = bias_initializer
        self.__input_length = input_length
        self.__phase_indicator = phase_indicator

    def call(self, input_x, prev_hidden_state):
        """
        :param input_x: BTD format
        :param prev_hidden_state:
        :return: new hidden state: BD format
        """
        input_length = self.__input_length
        num_hidden = self.__num_hidden
        weight_initializer = self.__weight_initializer
        bias_initializer = self.__bias_initializer
        with tf.variable_scope('DropGRU_Parameter', reuse=tf.AUTO_REUSE):
            weight_z = tf.get_variable('w_z', [num_hidden, input_length+num_hidden], initializer=weight_initializer)
            weight_r = tf.get_variable('w_r', [num_hidden, input_length+num_hidden], initializer=weight_initializer)
            weight_g = tf.get_variable('w_t', [num_hidden, input_length+num_hidden], initializer=weight_initializer)
            bias_z = tf.get_variable('b_z', [num_hidden, 1], initializer=bias_initializer)
            bias_r = tf.get_variable('b_r', [num_hidden, 1], initializer=bias_initializer)
            bias_g = tf.get_variable('b_t', [num_hidden, 1], initializer=bias_initializer)

        with tf.name_scope('DropGRU_Internal'):
            concat_1 = tf.transpose(tf.concat([input_x, prev_hidden_state], axis=1))
            z_t = tf.sigmoid(tf.matmul(weight_z, concat_1) + bias_z)
            r_t = tf.sigmoid(tf.matmul(weight_r, concat_1) + bias_r)

            concat_2 = tf.transpose(tf.concat([input_x, tf.transpose(r_t)*prev_hidden_state], axis=1))
            g_t = tf.matmul(weight_g, concat_2) + bias_g
            new_state = tf.transpose((1 - z_t) * tf.transpose(prev_hidden_state) + z_t * g_t)
        return new_state


class DropContextualGRUCell(object):
    def __init__(self, num_hidden, input_length, keep_prob, phase_indicator, weight_initializer, bias_initializer):
        """
        :param num_hidden:
        :param weight_initializer:
        :param bias_initializer:
        :param phase_indicator: phase_indicator>0代表是测试期，phase_indicator<=0代表是训练期
        :param keep_prob:
        """
        self.__num_hidden = num_hidden
        self.__weight_initializer = weight_initializer
        self.__bias_initializer = bias_initializer
        self.__input_length = input_length
        self.__keep_probability = keep_prob
        self.__phase_indicator = phase_indicator

    def call(self, input_x, prev_hidden_state):
        """
        :param input_x: BTD format
        :param prev_hidden_state:
        :return: new hidden state: BD format
        """
        keep_probability = self.__keep_probability
        input_length = self.__input_length
        num_hidden = self.__num_hidden
        weight_initializer = self.__weight_initializer
        bias_initializer = self.__bias_initializer
        phase_indicator = self.__phase_indicator

        with tf.variable_scope('DropGRU_Parameter', reuse=tf.AUTO_REUSE):
            weight_z = tf.get_variable('w_z', [num_hidden, input_length+num_hidden], initializer=weight_initializer)
            weight_r = tf.get_variable('w_r', [num_hidden, input_length+num_hidden], initializer=weight_initializer)
            weight_g = tf.get_variable('w_t', [num_hidden, input_length+num_hidden], initializer=weight_initializer)
            bias_z = tf.get_variable('b_z', [num_hidden, 1], initializer=bias_initializer)
            bias_r = tf.get_variable('b_r', [num_hidden, 1], initializer=bias_initializer)
            bias_g = tf.get_variable('b_t', [num_hidden, 1], initializer=bias_initializer)

        with tf.name_scope('DropGRU_Internal'):
            concat_1 = tf.transpose(tf.concat([input_x, prev_hidden_state], axis=1))
            z_t = tf.sigmoid(tf.matmul(weight_z, concat_1) + bias_z)
            r_t = tf.sigmoid(tf.matmul(weight_r, concat_1) + bias_r)

            concat_2 = tf.transpose(tf.concat([input_x, tf.transpose(r_t)*prev_hidden_state], axis=1))
            g_t = tf.matmul(weight_g, concat_2) + bias_g
            dropped_g_t = tf.cond(phase_indicator > 0,
                                  lambda: g_t,
                                  lambda: tf.nn.dropout(g_t, keep_probability))

            new_state = tf.transpose((1 - z_t) * tf.transpose(prev_hidden_state) + z_t * dropped_g_t)
        return new_state


def main():
    batch_size = tf.placeholder(tf.int32, [], name='batch_size')
    phase_indicator = tf.placeholder(tf.int32, shape=[], name="phase_indicator")
    num_hidden = 10
    num_steps = 20
    input_length = 7
    zero_state = tf.zeros([batch_size, num_hidden])
    initializer_o = tf.initializers.orthogonal()
    initializer_z = tf.initializers.zeros()
    c_cell = ContextualGRUCell(num_hidden=num_hidden, input_length=input_length, phase_indicator=phase_indicator,
                               weight_initializer=initializer_o, bias_initializer=initializer_z)
    d_cell = DropContextualGRUCell(num_hidden=num_hidden, input_length=input_length, phase_indicator=phase_indicator,
                                   weight_initializer=initializer_o, bias_initializer=initializer_z, keep_prob=0.9)

    x_placeholder = tf.placeholder(tf.float32, [None, num_steps, input_length], name='x_placeholder')
    x_unstack = tf.unstack(x_placeholder, axis=1)
    state = zero_state
    for i in range(num_steps):
        state = c_cell.call(x_unstack[i], state)
    for i in range(num_steps):
        state = d_cell.call(x_unstack[i], state)


if __name__ == '__main__':
    main()
