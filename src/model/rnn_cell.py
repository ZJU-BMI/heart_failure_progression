# encoding=utf-8-sig
import tensorflow as tf


class ContextualRawCell(object):
    """
    和标准的GRU Cell基本一致
    使用了arXiv 1603.05118 Recurrent Dropout without Memory Loss中所提及的对Hidden State的正则化策略
    """
    def __init__(self, num_hidden, event_number, context_number, keep_prob, phase_indicator, weight_initializer,
                 bias_initializer, activation=tf.tanh):
        """
        :param num_hidden:
        :param event_number:
        :param context_number:
        :param weight_initializer:
        :param bias_initializer:
        :param phase_indicator: phase_indicator>0代表是测试期，phase_indicator<=0代表是训练期
        :param keep_prob:
        """
        self.__num_hidden = num_hidden
        self.__weight_initializer = weight_initializer
        self.__bias_initializer = bias_initializer
        self.__event_number = event_number
        self.__context_number = context_number
        self.__keep_probability = keep_prob
        self.__phase_indicator = phase_indicator
        self.__variable_dict = None
        self.__activation = activation

        self.__build()

    def __call__(self, input_event, input_context, prev_hidden_state):
        keep_probability = self.__keep_probability
        phase_indicator = self.__phase_indicator
        weight = self.__variable_dict['weight']
        bias = self.__variable_dict['bias']
        activation = self.__activation

        with tf.name_scope('cell_internal'):
            concat_1 = tf.transpose(tf.concat([input_event, input_context, prev_hidden_state], axis=1))
            h_t = tf.transpose(activation(tf.matmul(weight, concat_1) + bias, name='hidden_state'))
            dropped_g_t = tf.cond(phase_indicator > 0,
                                  lambda: h_t,
                                  lambda: tf.nn.dropout(h_t, keep_probability))
            return dropped_g_t

    def __build(self):
        event_number = self.__event_number
        context_number = self.__context_number
        num_hidden = self.__num_hidden
        weight_initializer = self.__weight_initializer
        bias_initializer = self.__bias_initializer
        with tf.variable_scope('raw_rnn_cell_parameter'):
            weight = tf.get_variable('w', [num_hidden, event_number+context_number+num_hidden],
                                     initializer=weight_initializer)
            bias = tf.get_variable('b', [num_hidden, 1], initializer=bias_initializer)
        variable_dict = {'weight': weight, 'bias': bias}
        self.__variable_dict = variable_dict

    @property
    def num_hidden(self):
        return self.__num_hidden


class ContextualGRUCell(object):
    def __init__(self, num_hidden, event_number, context_number, keep_prob, phase_indicator, weight_initializer,
                 bias_initializer, activation=tf.tanh):
        """
        :param num_hidden:
        :param event_number:
        :param context_number:
        :param weight_initializer:
        :param bias_initializer:
        :param phase_indicator: phase_indicator>0代表是测试期，phase_indicator<=0代表是训练期
        :param keep_prob:
        """
        self.__num_hidden = num_hidden
        self.__weight_initializer = weight_initializer
        self.__bias_initializer = bias_initializer
        self.__event_number = event_number
        self.__context_number = context_number
        self.__keep_probability = keep_prob
        self.__phase_indicator = phase_indicator
        self.__variable_dict = None
        self.__activation = activation

        self.__build()

    def __call__(self, input_event, input_context, prev_hidden_state):
        """
        :param input_event: BTD format
        :param input_context: BTD format
        :param prev_hidden_state:
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

        with tf.name_scope('CGRU_Internal'):
            concat_1 = tf.transpose(tf.concat([input_event, input_context, prev_hidden_state], axis=1))
            z_t = tf.transpose(tf.sigmoid(tf.matmul(weight_z, concat_1) + bias_z))
            r_t = tf.transpose(tf.sigmoid(tf.matmul(weight_r, concat_1) + bias_r))

            concat_2 = tf.transpose(tf.concat([input_event, input_context, r_t*prev_hidden_state], axis=1))
            g_t = tf.transpose(activation(tf.matmul(weight_g, concat_2) + bias_g))

            dropped_g_t = tf.cond(phase_indicator > 0,
                                  lambda: g_t,
                                  lambda: tf.nn.dropout(g_t, keep_probability))

            new_state = (1 - z_t) * prev_hidden_state + z_t * dropped_g_t
        return new_state

    def __build(self):
        event_number = self.__event_number
        context_number = self.__context_number
        num_hidden = self.__num_hidden
        weight_initializer = self.__weight_initializer
        bias_initializer = self.__bias_initializer
        with tf.variable_scope('parameter'):
            weight_z = tf.get_variable('w_z', [num_hidden, event_number+context_number+num_hidden],
                                       initializer=weight_initializer)
            weight_r = tf.get_variable('w_r', [num_hidden, event_number+context_number+num_hidden],
                                       initializer=weight_initializer)
            weight_g = tf.get_variable('w_t', [num_hidden, event_number+context_number+num_hidden],
                                       initializer=weight_initializer)
            bias_z = tf.get_variable('b_z', [num_hidden, 1], initializer=bias_initializer)
            bias_r = tf.get_variable('b_r', [num_hidden, 1], initializer=bias_initializer)
            bias_g = tf.get_variable('b_t', [num_hidden, 1], initializer=bias_initializer)
        variable_dict = {'weight_z': weight_z, 'weight_r': weight_r, 'weight_g': weight_g, 'bias_z': bias_z,
                         'bias_r': bias_r, 'bias_g': bias_g}
        self.__variable_dict = variable_dict

    @property
    def num_hidden(self):
        return self.__num_hidden


class ContextualLSTMCell(object):
    def __init__(self, num_hidden, event_number, context_number, keep_prob, phase_indicator, weight_initializer,
                 bias_initializer, activation=tf.tanh):
        """
        :param num_hidden:
        :param event_number:
        :param context_number:
        :param weight_initializer:
        :param bias_initializer:
        :param phase_indicator: phase_indicator>0代表是测试期，phase_indicator<=0代表是训练期
        :param keep_prob:
        """
        self.__num_hidden = num_hidden
        self.__weight_initializer = weight_initializer
        self.__bias_initializer = bias_initializer
        self.__event_number = event_number
        self.__context_number = context_number
        self.__keep_probability = keep_prob
        self.__phase_indicator = phase_indicator
        self.__variable_dict = None
        self.__activation = activation

        self.__build()

    def __call__(self, input_event, input_context, prev_hidden_state, prev_cell_state):
        """
        :param input_event: BTD format
        :param input_context: BTD format
        :param prev_cell_state:
        :param prev_hidden_state:
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

        with tf.name_scope('CLSTM_Internal'):
            concat_1 = tf.transpose(tf.concat([input_event, input_context, prev_hidden_state], axis=1))
            f_t = tf.transpose(tf.sigmoid(tf.matmul(weight_f, concat_1) + bias_f))
            i_t = tf.transpose(tf.sigmoid(tf.matmul(weight_i, concat_1) + bias_i))
            o_t = tf.transpose(tf.sigmoid(tf.matmul(weight_o, concat_1) + bias_o))
            g_t = tf.transpose(self.__activation(tf.matmul(weight_c, concat_1) + bias_c))

            dropped_g_t = tf.cond(phase_indicator > 0,
                                  lambda: g_t,
                                  lambda: tf.nn.dropout(g_t, keep_probability))

            c_t = f_t*prev_cell_state + i_t*dropped_g_t
            h_t = o_t * self.__activation(c_t)
        return h_t, c_t

    def __build(self):
        event_number = self.__event_number
        context_number = self.__context_number
        num_hidden = self.__num_hidden
        weight_initializer = self.__weight_initializer
        bias_initializer = self.__bias_initializer
        with tf.variable_scope('parameter'):
            weight_f = tf.get_variable('w_f', [num_hidden, context_number+event_number+num_hidden],
                                       initializer=weight_initializer)
            weight_i = tf.get_variable('w_i', [num_hidden, context_number+event_number+num_hidden],
                                       initializer=weight_initializer)
            weight_o = tf.get_variable('w_o', [num_hidden, context_number+event_number+num_hidden],
                                       initializer=weight_initializer)
            weight_c = tf.get_variable('w_c', [num_hidden, context_number+event_number+num_hidden],
                                       initializer=weight_initializer)
            bias_f = tf.get_variable('b_f', [num_hidden, 1], initializer=bias_initializer)
            bias_i = tf.get_variable('b_i', [num_hidden, 1], initializer=bias_initializer)
            bias_o = tf.get_variable('b_o', [num_hidden, 1], initializer=bias_initializer)
            bias_c = tf.get_variable('b_c', [num_hidden, 1], initializer=bias_initializer)
        variable_dict = {'weight_f': weight_f, 'weight_i': weight_i, 'weight_o': weight_o, 'weight_c': weight_c,
                         'bias_f': bias_f, 'bias_i': bias_i, 'bias_o': bias_o, 'bias_c': bias_c}
        self.__variable_dict = variable_dict

    @property
    def num_hidden(self):
        return self.__num_hidden


def unit_test():
    """
    本函数用于断点测试，跟踪每个Cell中是否有错误
    :return:
    """
    batch_size = tf.placeholder(tf.int32, [], name='batch_size')
    num_hidden = 10
    num_steps = 20
    event_number = 25
    context_number = 30
    keep_prob = 1.0
    zero_state = tf.zeros([batch_size, num_hidden])
    initializer_o = tf.initializers.orthogonal()
    initializer_z = tf.initializers.zeros()

    # input should be BTD format
    event_input = tf.placeholder(tf.float32, [None, num_steps, event_number], name='event_input')
    context_input = tf.placeholder(tf.float32, [None, num_steps, context_number], name='context_input')
    phase_indicator = tf.placeholder(tf.int32, [], name='phase_indicator')

    # 试验阶段
    test_cell_type = 2
    event_input_item = tf.unstack(event_input, axis=1)
    context_input_item = tf.unstack(context_input, axis=1)
    if test_cell_type == 0:
        a_cell = ContextualGRUCell(num_hidden=num_hidden, context_number=context_number, event_number=event_number,
                                   weight_initializer=initializer_o, bias_initializer=initializer_z,
                                   keep_prob=keep_prob, phase_indicator=phase_indicator)
        state = zero_state
        state_list = list()
        for i in range(num_steps):
            # Contextual GRU Cell
            state = a_cell(input_event=event_input_item[i], input_context=context_input_item[i],
                           prev_hidden_state=state)
            state_list.append(state)
    elif test_cell_type == 1:
        b_cell = ContextualRawCell(num_hidden=num_hidden,  weight_initializer=initializer_o,
                                   bias_initializer=initializer_z, keep_prob=keep_prob, event_number=event_number,
                                   phase_indicator=phase_indicator, context_number=context_number)
        state = zero_state
        state_list = list()
        for i in range(num_steps):
            # Contextual GRU Hawkes Only
            state = b_cell(input_event=event_input_item[i], input_context=context_input_item[i],
                           prev_hidden_state=state)
            state_list.append(state)

    elif test_cell_type == 2:
        c_cell = ContextualLSTMCell(num_hidden=num_hidden, context_number=context_number, event_number=event_number,
                                    weight_initializer=initializer_o, bias_initializer=initializer_z,
                                    keep_prob=keep_prob, phase_indicator=phase_indicator)
        hidden_state = zero_state
        context_state = tf.zeros([batch_size, num_hidden])
        state_list = list()
        for i in range(num_steps):
            hidden_state, context_state = c_cell(input_event=event_input_item[i], input_context=context_input_item[i],
                                                 prev_hidden_state=hidden_state, prev_cell_state=context_state)
            state_list.append(hidden_state)
    else:
        print('No Cell Test')
    print('finish')


if __name__ == '__main__':
    unit_test()
