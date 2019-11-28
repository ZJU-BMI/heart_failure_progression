# encoding=utf-8-sig
# 本模块中使用的
import tensorflow as tf


class RawCell(object):
    """
    和标准的GRU Cell基本一致
    使用了arXiv 1603.05118 Recurrent Dropout without Memory Loss中所提及的对Hidden State的正则化策略
    """
    def __init__(self, num_hidden, input_length, keep_prob, phase_indicator, name,
                 weight_initializer=tf.initializers.orthogonal(),
                 bias_initializer=tf.initializers.zeros(),
                 activation=tf.tanh):
        """
        :param num_hidden:
        :param name: 之所以设定这个参数，是为了要创建多层rnn时，能够在计算图中区分cell，避免错误的reuse
        :param input_length:
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
        self.__variable_dict = None
        self.__activation = activation
        self.__name = name
        self.__build()

    def __call__(self, input_x, recurrent_state):
        keep_probability = self.__keep_probability
        phase_indicator = self.__phase_indicator
        weight = self.__variable_dict['weight']
        bias = self.__variable_dict['bias']
        activation = self.__activation

        with tf.name_scope('cell_internal'):
            concat_1 = tf.transpose(tf.concat([input_x, recurrent_state], axis=1))
            h_t = tf.transpose(activation(tf.matmul(weight, concat_1) + bias, name='hidden_state'))
            dropped_h_t = tf.cond(phase_indicator > 0,
                                  lambda: h_t,
                                  lambda: tf.nn.dropout(h_t, keep_probability))

        # 输出两个一模一样的结果是为了和Tensorflow保持一致
        # 之所以Tensorflow这么输出，猜测是因为LSTM要输出Hidden State和Cell State
        # 其它的Cell虽然没有Cell State这个设计，但是为了保证统一的编程框架，也要输出这么一个东西
        return dropped_h_t, dropped_h_t

    def __build(self):
        input_length = self.__input_length
        num_hidden = self.__num_hidden
        weight_initializer = self.__weight_initializer
        bias_initializer = self.__bias_initializer
        with tf.variable_scope('raw_para_'+self.__name):
            weight = tf.get_variable('w', [num_hidden, input_length+num_hidden],
                                     initializer=weight_initializer)
            bias = tf.get_variable('b', [num_hidden, 1], initializer=bias_initializer)
        variable_dict = {'weight': weight, 'bias': bias}
        self.__variable_dict = variable_dict

    @property
    def state_size(self):
        return self.__num_hidden

    @property
    def output_size(self):
        return self.__num_hidden

    def get_initial_state(self, batch_size, configure='zero'):
        if configure == 'zero':
            return tf.zeros([batch_size, self.__num_hidden])
        else:
            raise ValueError('')


class GRUCell(object):
    def __init__(self, num_hidden, input_length, keep_prob, phase_indicator, name,
                 weight_initializer=tf.initializers.orthogonal(),
                 bias_initializer=tf.initializers.zeros(),
                 activation=tf.tanh):
        """
        :param num_hidden:
        :param name: 之所以设定这个参数，是为了要创建多层rnn时，能够在计算图中区分cell，避免错误的reuse
        :param input_length:
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
        self.__variable_dict = None
        self.__activation = activation
        self.__name = name
        self.__build()

    def __call__(self, input_x, recurrent_state):
        """
        :param input_x: BD format
        :param recurrent_state:
        :return: new hidden state: BD format
        """
        keep_probability = self.__keep_probability
        phase_indicator = self.__phase_indicator
        activation = self.__activation

        variable_dict = self.__variable_dict
        weight_z = variable_dict['weight_z']
        weight_r = variable_dict['weight_r']
        weight_g = variable_dict['weight_g']
        bias_z = variable_dict['bias_z']
        bias_r = variable_dict['bias_r']
        bias_g = variable_dict['bias_g']

        with tf.name_scope('GRU_Internal'):
            concat_1 = tf.transpose(tf.concat([input_x, recurrent_state], axis=1))
            z_t = tf.transpose(tf.sigmoid(tf.matmul(weight_z, concat_1) + bias_z))
            r_t = tf.transpose(tf.sigmoid(tf.matmul(weight_r, concat_1) + bias_r))

            concat_2 = tf.transpose(tf.concat([input_x, r_t*recurrent_state], axis=1))
            g_t = tf.transpose(activation(tf.matmul(weight_g, concat_2) + bias_g))

            dropped_g_t = tf.cond(phase_indicator > 0,
                                  lambda: g_t,
                                  lambda: tf.nn.dropout(g_t, keep_probability))

            new_state = (1 - z_t) * recurrent_state + z_t * dropped_g_t
        return new_state, new_state

    def __build(self):
        input_length = self.__input_length
        num_hidden = self.__num_hidden
        weight_initializer = self.__weight_initializer
        bias_initializer = self.__bias_initializer
        with tf.variable_scope('gru_para_'+self.__name):
            weight_z = tf.get_variable('w_z', [num_hidden, input_length+num_hidden],
                                       initializer=weight_initializer)
            weight_r = tf.get_variable('w_r', [num_hidden, input_length+num_hidden],
                                       initializer=weight_initializer)
            weight_g = tf.get_variable('w_t', [num_hidden, input_length+num_hidden],
                                       initializer=weight_initializer)
            bias_z = tf.get_variable('b_z', [num_hidden, 1], initializer=bias_initializer)
            bias_r = tf.get_variable('b_r', [num_hidden, 1], initializer=bias_initializer)
            bias_g = tf.get_variable('b_t', [num_hidden, 1], initializer=bias_initializer)
        variable_dict = {'weight_z': weight_z, 'weight_r': weight_r, 'weight_g': weight_g, 'bias_z': bias_z,
                         'bias_r': bias_r, 'bias_g': bias_g}
        self.__variable_dict = variable_dict

    @property
    def state_size(self):
        return self.__num_hidden

    @property
    def output_size(self):
        return self.__num_hidden

    def get_initial_state(self, batch_size, configure='zero'):
        if configure == 'zero':
            return tf.zeros([batch_size, self.__num_hidden])
        else:
            raise ValueError('')


class LSTMCell(object):
    def __init__(self, num_hidden, input_length, keep_prob, phase_indicator, name,
                 weight_initializer=tf.initializers.orthogonal(),
                 bias_initializer=tf.initializers.zeros(),
                 activation=tf.tanh):
        """
        :param num_hidden:
        :param input_length:
        :param weight_initializer:
        :param bias_initializer:
        :param name: 之所以设定这个参数，是为了要创建多层rnn时，能够在计算图中区分cell，避免错误的reuse
        :param phase_indicator: phase_indicator>0代表是测试期，phase_indicator<=0代表是训练期
        :param keep_prob:
        """
        self.__num_hidden = num_hidden
        self.__weight_initializer = weight_initializer
        self.__bias_initializer = bias_initializer
        self.__input_length = input_length
        self.__keep_probability = keep_prob
        self.__phase_indicator = phase_indicator
        self.__variable_dict = None
        self.__activation = activation
        self.__name = name
        self.__build()

    def __call__(self, input_x, recurrent_state):
        """
        :param input_x: BD format
        :param recurrent_state: BD format
        :return: new hidden state: BD format
        """
        keep_probability = self.__keep_probability
        phase_indicator = self.__phase_indicator

        variable_dict = self.__variable_dict
        weight_f = variable_dict['weight_f']
        weight_i = variable_dict['weight_i']
        weight_o = variable_dict['weight_o']
        weight_c = variable_dict['weight_c']
        bias_f = variable_dict['bias_f']
        bias_i = variable_dict['bias_i']
        bias_o = variable_dict['bias_o']
        bias_c = variable_dict['bias_c']

        prev_hidden_state, prev_cell_state = tf.split(recurrent_state, 2, axis=1)
        with tf.name_scope('CLSTM_Internal'):
            concat_1 = tf.transpose(tf.concat([input_x, prev_hidden_state], axis=1))
            f_t = tf.transpose(tf.sigmoid(tf.matmul(weight_f, concat_1) + bias_f))
            i_t = tf.transpose(tf.sigmoid(tf.matmul(weight_i, concat_1) + bias_i))
            o_t = tf.transpose(tf.sigmoid(tf.matmul(weight_o, concat_1) + bias_o))
            g_t = tf.transpose(self.__activation(tf.matmul(weight_c, concat_1) + bias_c))

            dropped = tf.cond(phase_indicator > 0,
                              lambda: i_t*g_t,
                              lambda: tf.nn.dropout(i_t*g_t, keep_probability))

            c_t = f_t*prev_cell_state + dropped
            h_t = o_t * self.__activation(c_t)

            recurrent_state = tf.concat((h_t, c_t), axis=1)
        return h_t, recurrent_state

    def __build(self):
        input_length = self.__input_length
        num_hidden = self.__num_hidden
        weight_initializer = self.__weight_initializer
        bias_initializer = self.__bias_initializer
        with tf.variable_scope('lstm_para_'+self.__name):
            weight_f = tf.get_variable('w_f', [num_hidden, input_length+num_hidden],
                                       initializer=weight_initializer)
            weight_i = tf.get_variable('w_i', [num_hidden, input_length+num_hidden],
                                       initializer=weight_initializer)
            weight_o = tf.get_variable('w_o', [num_hidden, input_length+num_hidden],
                                       initializer=weight_initializer)
            weight_c = tf.get_variable('w_c', [num_hidden, input_length+num_hidden],
                                       initializer=weight_initializer)
            bias_f = tf.get_variable('b_f', [num_hidden, 1], initializer=bias_initializer)
            bias_i = tf.get_variable('b_i', [num_hidden, 1], initializer=bias_initializer)
            bias_o = tf.get_variable('b_o', [num_hidden, 1], initializer=bias_initializer)
            bias_c = tf.get_variable('b_c', [num_hidden, 1], initializer=bias_initializer)
        variable_dict = {'weight_f': weight_f, 'weight_i': weight_i, 'weight_o': weight_o, 'weight_c': weight_c,
                         'bias_f': bias_f, 'bias_i': bias_i, 'bias_o': bias_o, 'bias_c': bias_c}
        self.__variable_dict = variable_dict

    @property
    def state_size(self):
        return self.__num_hidden

    @property
    def output_size(self):
        return self.__num_hidden

    def get_initial_state(self, batch_size, configure='zero'):
        if configure == 'zero':
            return tf.zeros([batch_size, self.__num_hidden*2])
        else:
            raise ValueError('')


def unit_test():
    """
    本函数用于断点测试，跟踪每个Cell中是否有错误
    :return:
    """
    batch_size = tf.placeholder(tf.int32, [], name='batch_size')
    num_hidden = 10
    num_steps = 20
    input_length = 25
    keep_prob = 1.0
    zero_state = tf.zeros([batch_size, num_hidden])
    initializer_o = tf.initializers.orthogonal()
    initializer_z = tf.initializers.zeros()

    # input should be TBD format
    input_x = tf.placeholder(tf.float32, [num_steps, None, input_length], name='event_input')
    phase_indicator = tf.placeholder(tf.int32, [], name='phase_indicator')

    # 试验阶段
    test_cell_type = 0
    input_x_list = tf.unstack(input_x, axis=0)
    if test_cell_type == 0:
        a_cell = GRUCell(num_hidden=num_hidden, input_length=input_length, weight_initializer=initializer_o,
                         bias_initializer=initializer_z, keep_prob=keep_prob, phase_indicator=phase_indicator,
                         name='gru')
        recurrent_state = zero_state
        state_list = list()
        for i in range(num_steps):
            output_state, recurrent_state = a_cell(input_x=input_x_list[i], recurrent_state=recurrent_state)
            state_list.append(output_state)
    elif test_cell_type == 1:
        b_cell = RawCell(num_hidden=num_hidden,  weight_initializer=initializer_o, bias_initializer=initializer_z,
                         keep_prob=keep_prob, input_length=input_length, phase_indicator=phase_indicator,
                         name='raw')
        recurrent_state = zero_state
        state_list = list()
        for i in range(num_steps):
            output_state, recurrent_state = b_cell(input_x=input_x_list[i], recurrent_state=recurrent_state)
            state_list.append(output_state)

    elif test_cell_type == 2:
        c_cell = LSTMCell(num_hidden=num_hidden, input_length=input_length,weight_initializer=initializer_o,
                          bias_initializer=initializer_z, keep_prob=keep_prob, phase_indicator=phase_indicator,
                          name='lstm')
        hidden_state = zero_state
        context_state = tf.zeros([batch_size, num_hidden])
        recurrent_state = tf.concat([hidden_state, context_state], axis=1)
        state_list = list()
        for i in range(num_steps):
            output_state, recurrent_state = c_cell(input_x=input_x_list[i], recurrent_state=recurrent_state)
            state_list.append(output_state)
    else:
        print('No Cell Test')
    print('finish')


if __name__ == '__main__':
    unit_test()
