import tensorflow as tf
# 此处虽然定义了ContextualGRUCell的定义
# 但是Contextual GRU其实原版GRU在算法上没有区别，其Contextual更像是在语义上进行定义


class ContextualGRUCell(object):
    def __init__(self, num_hidden, input_length, phase_indicator, weight_initializer, bias_initializer,
                 initial_hidden_state):
        """
        :param num_hidden:
        :param weight_initializer:
        :param bias_initializer:
        :param phase_indicator: phase_indicator>0代表是测试期，phase_indicator<=0代表是训练期
        :param initial_hidden_state: 和TF的基础设置一样，默认Hidden_State初始化为0
        """
        self.__num_hidden = num_hidden
        self.__weight_initializer = weight_initializer
        self.__bias_initializer = bias_initializer
        self.__input_length = input_length
        self.__initial_hidden_state = initial_hidden_state
        self.__phase_indicator = phase_indicator

    def call(self, input_x, prev_hidden_state):
        input_length = self.__input_length
        num_hidden = self.__num_hidden
        weight_initializer = self.__weight_initializer
        bias_initializer = self.__bias_initializer
        with tf.variable_scope('DropGRU_Parameter', reuse=tf.AUTO_REUSE):
            weight_z = tf.get_variable('w_z', [input_length+num_hidden, num_hidden], initializer=weight_initializer)
            weight_r = tf.get_variable('w_r', [input_length+num_hidden, num_hidden], initializer=weight_initializer)
            weight_g = tf.get_variable('w_t', [input_length+num_hidden, num_hidden], initializer=weight_initializer)
            bias_z = tf.get_variable('b_z', [num_hidden, ], initializer=bias_initializer)
            bias_r = tf.get_variable('b_r', [num_hidden, ], initializer=bias_initializer)
            bias_g = tf.get_variable('b_t', [num_hidden, ], initializer=bias_initializer)

        with tf.name_scope('DropGRU_Internal'):
            z_t = tf.matmul(tf.concat([input_x, prev_hidden_state], axis=1), weight_z) + bias_z
            z_t = tf.sigmoid(z_t)
            r_t = tf.matmul(tf.concat([input_x, prev_hidden_state], axis=1), weight_r) + bias_r
            r_t = tf.sigmoid(r_t)
            g_t = tf.matmul(tf.concat([input_x, r_t*prev_hidden_state], axis=1), weight_g) + bias_g
            new_state = (1 - z_t) * prev_hidden_state + z_t * g_t
        return new_state


class DropContextualGRUCell(object):
    def __init__(self, num_hidden, input_length, keep_prob, phase_indicator, weight_initializer, bias_initializer,
                 initial_hidden_state):
        """
        :param num_hidden:
        :param weight_initializer:
        :param bias_initializer:
        :param phase_indicator: phase_indicator>0代表是测试期，phase_indicator<=0代表是训练期
        :param keep_prob:
        :param initial_hidden_state: 和TF的基础设置一样，默认Hidden_State初始化为0
        """
        self.__num_hidden = num_hidden
        self.__weight_initializer = weight_initializer
        self.__bias_initializer = bias_initializer
        self.__input_length = input_length
        self.__keep_probability = keep_prob
        self.__initial_hidden_state = initial_hidden_state
        self.__phase_indicator = phase_indicator

    def call(self, input_x, prev_hidden_state):
        keep_probability = self.__keep_probability
        input_length = self.__input_length
        num_hidden = self.__num_hidden
        weight_initializer = self.__weight_initializer
        bias_initializer = self.__bias_initializer
        phase_indicator = self.__phase_indicator

        with tf.variable_scope('DropGRU_Parameter', reuse=tf.AUTO_REUSE):
            weight_z = tf.get_variable('w_z', [input_length+num_hidden, num_hidden], initializer=weight_initializer)
            weight_r = tf.get_variable('w_r', [input_length+num_hidden, num_hidden], initializer=weight_initializer)
            weight_g = tf.get_variable('w_t', [input_length+num_hidden, num_hidden], initializer=weight_initializer)
            bias_z = tf.get_variable('b_z', [num_hidden, ], initializer=bias_initializer)
            bias_r = tf.get_variable('b_r', [num_hidden, ], initializer=bias_initializer)
            bias_g = tf.get_variable('b_t', [num_hidden, ], initializer=bias_initializer)

        with tf.name_scope('DropGRU_Internal'):
            z_t = tf.sigmoid(tf.matmul(tf.concat([input_x, prev_hidden_state], axis=1), weight_z) + bias_z)
            r_t = tf.sigmoid(tf.matmul(tf.concat([input_x, prev_hidden_state], axis=1), weight_r) + bias_r)
            g_t = tf.tanh(tf.matmul(tf.concat([input_x, r_t*prev_hidden_state], axis=1), weight_g) + bias_g)
            dropped_g_t = tf.cond(phase_indicator > 0,
                                  lambda: g_t,
                                  lambda: tf.nn.dropout(g_t, keep_probability))
            new_state = (1 - z_t) * prev_hidden_state + z_t * dropped_g_t
        return new_state
