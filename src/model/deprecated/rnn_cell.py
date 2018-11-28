import tensorflow as tf
import numpy as np
# 定义了实验有可能会用到的5种Cell
# Unit Test方法对Cell逐步进行了变量校验，目前没发现什么显著错误
# 所谓Contextual GRU的概念借鉴于Contextual LSTM（https://arxiv.org/abs/1602.06291）
# 在计算细节上和原版GRU没有任何区别，在本实验中，Contextual信息指代检查检验结果，用药等对病人病情的描述性信息


class ContextualGRUCellTimeDecay(object):
    """
    普通C-GRU，完全使用时间作为门控函数（准确的说是exp(-0.006*delta t), -0.006无特殊意义，仅仅是为了保证和Hawkes参数相同）
    """
    def __init__(self, num_hidden, input_length, weight_initializer, bias_initializer, omega=0.006):
        """
        :param num_hidden:
        :param input_length:
        :param omega:
        :param weight_initializer:
        :param bias_initializer:
        """
        self.__num_hidden = num_hidden
        self.__weight_initializer = weight_initializer
        self.__bias_initializer = bias_initializer
        self.__input_length = input_length
        self.__omega = omega

    def __call__(self, input_x, prev_hidden_state, time_interval):
        """
        :param input_x: BD format, D implies input length
        :param prev_hidden_state:
        :param time_interval: BD format, D=1, 本次入院距离上次入院的时间长度，若为第一次入院，则置0
        :return: BD format, D implies num hidden
        """
        input_length = self.__input_length
        num_hidden = self.__num_hidden
        weight_initializer = self.__weight_initializer
        bias_initializer = self.__bias_initializer
        omega = self.__omega

        # 基于上述强度指导三个权重的分配

        with tf.variable_scope('CGRU_Time_Parameter', reuse=tf.AUTO_REUSE):
            weight_z = tf.get_variable('w_z', [num_hidden, 1], initializer=weight_initializer)
            weight_r = tf.get_variable('w_r', [num_hidden, 1], initializer=weight_initializer)
            weight_g = tf.get_variable('w_t', [num_hidden, num_hidden+input_length], initializer=weight_initializer)
            bias_z = tf.get_variable('b_z', [num_hidden, 1], initializer=bias_initializer)
            bias_r = tf.get_variable('b_r', [num_hidden, 1], initializer=bias_initializer)
            bias_g = tf.get_variable('b_t', [num_hidden, 1], initializer=bias_initializer)

        with tf.name_scope('CGRU_Time_Cell'):
            with tf.name_scope('get_intensity'):
                time_interval = tf.expand_dims(time_interval, axis=1)
                time_intensity = tf.transpose(tf.exp(-omega * time_interval))
            with tf.name_scope('CGRU_Time_Internal'):
                z_t = tf.transpose(tf.sigmoid(tf.matmul(weight_z, time_intensity) + bias_z))
                r_t = tf.transpose(tf.sigmoid(tf.matmul(weight_r, time_intensity) + bias_r))

                concat_2 = tf.transpose(tf.concat([input_x, r_t*prev_hidden_state], axis=1))
                g_t = tf.transpose(tf.tanh(tf.matmul(weight_g, concat_2) + bias_g))
                new_state = z_t * prev_hidden_state + (1 - z_t) * g_t
        return new_state


class ContextualGRUHawkes(object):
    """
    普通C-GRU，使用Hawkes过程指导下的时间衰减变换函数
    """
    def __init__(self, num_hidden, input_length, base_intensity, mutual_intensity, weight_initializer,
                 bias_initializer, event_count=11, omega=0.006):
        """
        :param num_hidden:
        :param weight_initializer:
        :param bias_initializer:
        :param input_length:
        :param mutual_intensity:
        :param base_intensity:
        :param event_count:
        :param omega:
        """
        self.__num_hidden = num_hidden
        self.__weight_initializer = weight_initializer
        self.__bias_initializer = bias_initializer
        self.__input_length = input_length
        self.__event_count = event_count
        self.__omega = omega
        self.__base_intensity = base_intensity
        self.__mutual_intensity = mutual_intensity

        with tf.variable_scope('CGRU_Hawkes_Parameter', reuse=tf.AUTO_REUSE):
            self.weight_z = tf.get_variable('w_z', [num_hidden, 1], initializer=weight_initializer)
            self.weight_r = tf.get_variable('w_r', [num_hidden, 1], initializer=weight_initializer)
            self.weight_g = \
                tf.get_variable('w_t', [num_hidden, num_hidden + input_length], initializer=weight_initializer)
            self.bias_z = tf.get_variable('b_z', [num_hidden, 1], initializer=bias_initializer)
            self.bias_r = tf.get_variable('b_r', [num_hidden, 1], initializer=bias_initializer)
            self.bias_g = tf.get_variable('b_t', [num_hidden, 1], initializer=bias_initializer)

    def __call__(self, input_x, previous_input, prev_hidden_state,  task_index, time_interval):
        """
        :param input_x: BD format, D implies input length
        :param prev_hidden_state:
        :param previous_input:
        :param time_interval: 本次入院距离上次入院的时间长度，若为第一次入院，则置0
        :param task_index: 当前预测事件所对应的index，这一index需全局统一

        :return: BD format, D implies num of hidden
        """
        omega = self.__omega
        event_count = self.__event_count
        base_intensity = self.__base_intensity
        mutual_intensity = self.__mutual_intensity

        with tf.name_scope('CGRUHawkes_Cell'):
            with tf.name_scope('get_intensity'):
                event_list = tf.slice(previous_input, [0, 0], [-1, event_count])

                # 获取互激发矩阵强度
                # 互激发矩阵中的元素e_ij指代第j类事件激发第i类事件的强度
                base_intensity = tf.convert_to_tensor(base_intensity, dtype=tf.float32)
                mutual_intensity = tf.convert_to_tensor(mutual_intensity, dtype=tf.float32)
                mutual_intensity = mutual_intensity[task_index, :]
                mutual_intensity = tf.expand_dims(mutual_intensity, axis=1)
                mutual_intensity = tf.matmul(event_list, mutual_intensity)

                # 基于上述强度指导三个权重的分配
                time_interval = tf.expand_dims(time_interval, axis=1)
                time_intensity = tf.exp(-omega * time_interval)
                hawkes_intensity = tf.transpose(time_intensity * mutual_intensity + base_intensity[task_index])

            with tf.name_scope('CGRU_Hawkes_Internal'):
                z_t = tf.transpose(tf.sigmoid(tf.matmul(self.weight_z, hawkes_intensity) + self.bias_z))
                r_t = tf.transpose(tf.sigmoid(tf.matmul(self.weight_r, hawkes_intensity) + self.bias_r))

                concat_2 = tf.transpose(tf.concat([input_x, r_t*prev_hidden_state], axis=1))
                g_t = tf.transpose(tf.tanh(tf.matmul(self.weight_g, concat_2) + self.bias_g))
                new_state = z_t * prev_hidden_state + (1 - z_t) * g_t
        return new_state


class ContextualGRUCell(object):
    """
    最为基本的C-GRU
    """
    def __init__(self, num_hidden, input_length, weight_initializer, bias_initializer):
        """
        :param num_hidden:
        :param weight_initializer:
        :param bias_initializer:
        """
        self.__num_hidden = num_hidden
        self.__weight_initializer = weight_initializer
        self.__bias_initializer = bias_initializer
        self.__input_length = input_length

    def __call__(self, input_x, prev_hidden_state):
        """
        :param input_x: BD format, D implies input length
        :param prev_hidden_state:
        :return: new hidden state: BD format, D implies num hidden
        """
        input_length = self.__input_length
        num_hidden = self.__num_hidden
        weight_initializer = self.__weight_initializer
        bias_initializer = self.__bias_initializer
        with tf.variable_scope('CGRU_Parameter', reuse=tf.AUTO_REUSE):
            weight_z = tf.get_variable('w_z', [num_hidden, input_length+num_hidden], initializer=weight_initializer)
            weight_r = tf.get_variable('w_r', [num_hidden, input_length+num_hidden], initializer=weight_initializer)
            weight_g = tf.get_variable('w_t', [num_hidden, input_length+num_hidden], initializer=weight_initializer)
            bias_z = tf.get_variable('b_z', [num_hidden, 1], initializer=bias_initializer)
            bias_r = tf.get_variable('b_r', [num_hidden, 1], initializer=bias_initializer)
            bias_g = tf.get_variable('b_t', [num_hidden, 1], initializer=bias_initializer)

        with tf.name_scope('CGRU_Cell'):
            concat_1 = tf.transpose(tf.concat([input_x, prev_hidden_state], axis=1))
            z_t = tf.transpose(tf.sigmoid(tf.matmul(weight_z, concat_1) + bias_z))
            r_t = tf.transpose(tf.sigmoid(tf.matmul(weight_r, concat_1) + bias_r))

            concat_2 = tf.transpose(tf.concat([input_x, r_t*prev_hidden_state], axis=1))
            g_t = tf.transpose(tf.tanh(tf.matmul(weight_g, concat_2) + bias_g))
            new_state = z_t * prev_hidden_state + (1 - z_t) * g_t
        return new_state


class ContextualGRUCellFuseTimeDecay(object):
    """
    在C-GRU的基础上，在隐藏层的数据中整合加入时间衰减效应对门控的影响
    """
    def __init__(self, num_hidden, input_length, weight_initializer, bias_initializer, omega=0.006):
        """
        :param num_hidden:
        :param weight_initializer:
        :param bias_initializer:
        """
        self.__num_hidden = num_hidden
        self.__weight_initializer = weight_initializer
        self.__bias_initializer = bias_initializer
        self.__input_length = input_length
        self.__omega = omega

    def __call__(self, input_x, prev_hidden_state, time_interval):
        """
        :param input_x: BD format
        :param prev_hidden_state:
        :param time_interval:
        :return: new hidden state: BD format, D implies
        """
        input_length = self.__input_length
        num_hidden = self.__num_hidden
        weight_initializer = self.__weight_initializer
        bias_initializer = self.__bias_initializer
        omega = self.__omega

        with tf.variable_scope('CGRU_Fuse_Time_Parameter', reuse=tf.AUTO_REUSE):
            # 最后加的1是时间维度的映射
            weight_z = tf.get_variable('w_z', [num_hidden, input_length+num_hidden+1], initializer=weight_initializer)
            weight_r = tf.get_variable('w_r', [num_hidden, input_length+num_hidden+1], initializer=weight_initializer)
            weight_g = tf.get_variable('w_t', [num_hidden, input_length+num_hidden+1], initializer=weight_initializer)
            bias_z = tf.get_variable('b_z', [num_hidden, 1], initializer=bias_initializer)
            bias_r = tf.get_variable('b_r', [num_hidden, 1], initializer=bias_initializer)
            bias_g = tf.get_variable('b_t', [num_hidden, 1], initializer=bias_initializer)

        with tf.name_scope('CGRU_Fuse_Time_Cell'):
            with tf.name_scope('get_time_intensity'):
                time_interval = tf.expand_dims(time_interval, axis=1)
                time_intensity = tf.exp(-omega * time_interval)
            with tf.name_scope('CGRU_Fuse_Time_Internal'):
                concat_1 = tf.transpose(tf.concat([input_x, prev_hidden_state, time_intensity], axis=1))
                z_t = tf.transpose(tf.sigmoid(tf.matmul(weight_z, concat_1) + bias_z))
                r_t = tf.transpose(tf.sigmoid(tf.matmul(weight_r, concat_1) + bias_r))

                concat_2 = tf.transpose(tf.concat([input_x, r_t * prev_hidden_state, time_intensity], axis=1))
                g_t = tf.transpose(tf.tanh(tf.matmul(weight_g, concat_2) + bias_g))
                new_state = z_t * prev_hidden_state + (1 - z_t) * g_t
        return new_state


class ContextualGRUCellFuseHawkes(object):
    """
    在C-GRU的基础上，在隐藏层的数据中整合加入时间衰减效应对门控的影响
    """
    def __init__(self, num_hidden, input_length, weight_initializer, bias_initializer, mutual_intensity, base_intensity,
                 omega=0.006, event_count=11):
        """
        :param num_hidden:
        :param weight_initializer:
        :param input_length:
        :param mutual_intensity:
        :param base_intensity:
        :param omega:
        :param event_count:
        :param bias_initializer:
        """
        self.__num_hidden = num_hidden
        self.__weight_initializer = weight_initializer
        self.__bias_initializer = bias_initializer
        self.__input_length = input_length
        self.__mutual_intensity = mutual_intensity
        self.__base_intensity = base_intensity
        self.__event_count = event_count
        self.__omega = omega

    def __call__(self, input_x, previous_input, prev_hidden_state, task_index, time_interval):
        """
        :param input_x: BD format
        :param prev_hidden_state:
        :param time_interval:
        :param previous_input:
        :param task_index:
        :return: new hidden state: BD format, D implies num hidden unit
        """
        input_length = self.__input_length
        num_hidden = self.__num_hidden
        weight_initializer = self.__weight_initializer
        bias_initializer = self.__bias_initializer
        omega = self.__omega
        event_count = self.__event_count
        base_intensity = self.__base_intensity
        mutual_intensity = self.__mutual_intensity

        with tf.variable_scope('CGRU_Fuse_Hawkes_Parameter', reuse=tf.AUTO_REUSE):
            # 最后加的1是时间维度的映射
            weight_z = tf.get_variable('w_z', [num_hidden, input_length+num_hidden+1], initializer=weight_initializer)
            weight_r = tf.get_variable('w_r', [num_hidden, input_length+num_hidden+1], initializer=weight_initializer)
            weight_g = tf.get_variable('w_t', [num_hidden, input_length+num_hidden+1], initializer=weight_initializer)
            bias_z = tf.get_variable('b_z', [num_hidden, 1], initializer=bias_initializer)
            bias_r = tf.get_variable('b_r', [num_hidden, 1], initializer=bias_initializer)
            bias_g = tf.get_variable('b_t', [num_hidden, 1], initializer=bias_initializer)

        with tf.name_scope('CGRU_Fuse_Hawkes_Cell'):
            with tf.name_scope('get_time_intensity'):
                event_list = tf.slice(previous_input, [0, 0], [-1, event_count])

                # 获取互激发矩阵强度
                # 互激发矩阵中的元素e_ij指代第j类事件激发第i类事件的强度
                base_intensity = tf.convert_to_tensor(base_intensity, dtype=tf.float32)
                mutual_intensity = tf.convert_to_tensor(mutual_intensity, dtype=tf.float32)
                mutual_intensity = mutual_intensity[task_index, :]
                mutual_intensity = tf.expand_dims(mutual_intensity, axis=1)
                mutual_intensity = tf.matmul(event_list, mutual_intensity)

                # 基于上述强度指导三个权重的分配
                time_interval = tf.expand_dims(time_interval, axis=1)
                time_intensity = tf.exp(-omega * time_interval)
                hawkes_intensity = time_intensity * mutual_intensity + base_intensity[task_index]
            with tf.name_scope('CGRU_Fuse_Hawkes_Internal'):
                concat_1 = tf.transpose(tf.concat([input_x, prev_hidden_state, hawkes_intensity], axis=1))
                z_t = tf.transpose(tf.sigmoid(tf.matmul(weight_z, concat_1) + bias_z))
                r_t = tf.transpose(tf.sigmoid(tf.matmul(weight_r, concat_1) + bias_r))

                concat_2 = tf.transpose(tf.concat([input_x, r_t * prev_hidden_state, time_intensity], axis=1))
                g_t = tf.transpose(tf.tanh(tf.matmul(weight_g, concat_2) + bias_g))
                new_state = z_t * prev_hidden_state + (1 - z_t) * g_t
        return new_state


def unit_test():
    """
    本函数用于断点测试，跟踪每个Cell中是否有错误
    :return:
    """
    batch_size = tf.placeholder(tf.int32, [], name='batch_size')
    num_hidden = 10
    num_steps = 20
    input_length = 25
    zero_state = tf.zeros([batch_size, num_hidden])
    initializer_o = tf.initializers.orthogonal()
    initializer_z = tf.initializers.zeros()
    omega = 0.006
    event_count = 11
    mutual_intensity_matrix = np.random.uniform(0, 1, [event_count, event_count])
    base_intensity_vector = np.random.uniform(0, 1, [event_count, 1])

    # input should be BTD format
    x_placeholder = tf.placeholder(tf.float32, [None, num_steps, input_length], name='x_placeholder')
    time_interval = tf.placeholder(tf.float32, [None, num_steps], name='time_interval')
    task_index = tf.placeholder(tf.int32, [], name='time_interval')

    # 试验阶段
    test_cell_type = 4
    x_unstack = tf.unstack(x_placeholder, axis=1)
    time_unstack = tf.unstack(time_interval, axis=1)
    if test_cell_type == 0:
        a_cell = ContextualGRUCell(num_hidden=num_hidden, input_length=input_length,
                                   weight_initializer=initializer_o, bias_initializer=initializer_z)
        state = zero_state
        state_list = list()
        for i in range(num_steps):
            # Contextual GRU Cell
            state = a_cell(input_x=x_unstack[i], prev_hidden_state=state)
            state_list.append(state)

    elif test_cell_type == 1:
        b_cell = ContextualGRUHawkes(num_hidden=num_hidden, input_length=input_length, omega=omega,
                                     event_count=event_count, weight_initializer=initializer_o,
                                     bias_initializer=initializer_z, mutual_intensity=mutual_intensity_matrix,
                                     base_intensity=base_intensity_vector)
        state = zero_state
        state_list = list()
        for i in range(num_steps):
            # Contextual GRU Hawkes Only
            state = b_cell(input_x=x_unstack[i], prev_hidden_state=state, task_index=task_index,
                           time_interval=time_unstack[i], previous_input=x_unstack[i])
            state_list.append(state)

    elif test_cell_type == 2:
        c_cell = ContextualGRUCellTimeDecay(num_hidden=num_hidden, input_length=input_length,
                                            weight_initializer=initializer_o, bias_initializer=initializer_z)
        state = zero_state
        state_list = list()
        for i in range(num_steps):
            # Contextual GRU Cell Direct Time Decay
            state = c_cell(input_x=x_unstack[i], prev_hidden_state=state, time_interval=time_unstack[i])
            state_list.append(state)

    elif test_cell_type == 3:
        d_cell = ContextualGRUCellFuseTimeDecay(num_hidden=num_hidden, input_length=input_length,
                                                weight_initializer=initializer_o, bias_initializer=initializer_z)
        state = zero_state
        state_list = list()
        for i in range(num_steps):
            # Contextual GRU Cell Fuse Time Decay
            state = d_cell(input_x=x_unstack[i], prev_hidden_state=state, time_interval=time_unstack[i])
            state_list.append(state)

    elif test_cell_type == 4:
        e_cell = ContextualGRUCellFuseHawkes(num_hidden=num_hidden, input_length=input_length, omega=omega,
                                             event_count=event_count, weight_initializer=initializer_o,
                                             bias_initializer=initializer_z, mutual_intensity=mutual_intensity_matrix,
                                             base_intensity=base_intensity_vector)
        state = zero_state
        state_list = list()
        for i in range(num_steps):
            # Contextual GRU Fuse Hawkes
            state = e_cell(input_x=x_unstack[i], prev_hidden_state=state, task_index=task_index,
                           time_interval=time_unstack[i], previous_input=x_unstack[i])
            state_list.append(state)

    else:
        print('No Cell Test')


if __name__ == '__main__':
    unit_test()


# deprecated
class DropContextualGRUCell(object):
    """
    使用C-GRU，使用了arXiv 1603.05118 Recurrent Dropout without Memory Loss中所提及的正则化策略
    随机丢弃隐层数据，因此结构和上述略有不同。但是这种结构在实际实验中被证明效果不佳
    最后实验中没有用这种策略
    """
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

    def __call__(self, input_x, prev_hidden_state):
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
