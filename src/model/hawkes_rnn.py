# encoding=utf-8-sig
import tensorflow as tf
from rnn_cell import ContextualGRUCell
from rnn_cell import ContextualLSTMCell
from rnn_cell import ContextualRawCell


def __hawkes_rnn(rnn_cell, num_steps, x_placeholder, batch_size, base_intensity, event_list,
                 mutual_intensity, task_type, time_interval, markov_assumption):
    """
    :param num_steps:
    :param rnn_cell:
    :param x_placeholder:
    :param batch_size:
    :param base_intensity:
    :param mutual_intensity:
    :param time_interval:
    :param task_type:
    :return: loss, prediction, x_placeholder, y_placeholder, batch_size, phase_indicator
    其中 phase_indicator>0代表是测试期，<=0代表是训练期
    """
    num_hidden = rnn_cell.num_hidden
    zero_state = tf.zeros([batch_size, num_hidden])

    output_list = list()
    x_unstack = tf.unstack(x_placeholder, axis=1)
    time_interval = tf.unstack(time_interval, axis=1)
    state = zero_state

    for i in range(num_steps):
        if i == 0:
            state = rnn_cell(x_unstack[i], x_unstack[i], state, task_type, time_interval[i])
        else:
            state = rnn_cell(x_unstack[i], x_unstack[i-1], state, task_type, time_interval[i])
        output_list.append(state)
    return output_list


def hawkes_rnn_model(cell, num_steps, num_hidden, num_context, num_event, keep_rate_input, markov_assumption=True,
                     event_count=11, auto_encoder_value=15, auto_encoder_initializer=tf.initializers.orthogonal()):
    """
    :param cell:
    :param num_steps:
    :param num_hidden:
    :param num_context:
     :param num_event:
    :param keep_rate_input:
    :param event_count:
    :param markov_assumption:
    :param auto_encoder_value: 大于0时执行对输入的自编码，值即为最终降到的维度
    :param auto_encoder_initializer:
    :return:
    loss, prediction, x_placeholder, y_placeholder, batch_size, phase_indicator
    其中 phase_indicator>0代表是测试期，<=0代表是训练期
    """
    with tf.name_scope('hawkes_rnn'):
        with tf.name_scope('data_source'):
            batch_size = tf.placeholder(tf.int32, [], name='batch_size')
            # 标准输入规定为BTD
            event_placeholder = tf.placeholder(tf.float32, [None, num_steps, num_event], name='event_placeholder')
            context_placeholder = tf.placeholder(tf.float32, [None, num_steps, num_context], name='context_placeholder')
            y_placeholder = tf.placeholder(tf.float32, [None, 1], name='y_placeholder')
            phase_indicator = tf.placeholder(tf.int32, shape=[], name="phase_indicator")
            task_type = tf.placeholder(tf.int32, shape=[], name="task_type")
            mutual_intensity = tf.placeholder(tf.float32, shape=[event_count, event_count], name="mutual_intensity")
            base_intensity = tf.placeholder(tf.float32, shape=[event_count, 1], name="base_intensity")
            time_interval = tf.placeholder(tf.float32, shape=[None, num_steps], name="time_interval")

        with tf.name_scope('rnn'):

            # 用于判断是训练阶段还是测试阶段，用于判断是否要加Dropout
            # dropout只加在context信息上
            context_dropout = tf.cond(phase_indicator > 0,
                                      lambda: context_placeholder,
                                      lambda: tf.nn.dropout(context_placeholder, keep_rate_input))
            input_x = tf.concat([event_placeholder, context_dropout], axis=2)

            if auto_encoder_value > 0:
                auto_encoder = tf.get_variable('auto_encoder', [num_context+num_event, auto_encoder_value],
                                               initializer=auto_encoder_initializer)
                unstacked_list = tf.unstack(input_x, axis=1)
                coded_list = list()
                for single_input in unstacked_list:
                    coded_list.append(tf.transpose(tf.matmul(single_input, auto_encoder)))
                processed_input = tf.convert_to_tensor(coded_list)
            else:
                processed_input = input_x

            # 确保输入格式为 BTD
            processed_input = tf.transpose(processed_input, [2, 0, 1])
            output_list = __hawkes_rnn(base_intensity=base_intensity, mutual_intensity=mutual_intensity,
                                       batch_size=batch_size, num_steps=num_steps, task_type=task_type,
                                       rnn_cell=cell, x_placeholder=processed_input, time_interval=time_interval,
                                       markov_assumption=markov_assumption, event_list=event_placeholder)

    with tf.variable_scope('output_layer'):
        output_weight = tf.get_variable("weight", [num_hidden, 1], initializer=tf.initializers.orthogonal())
        bias = tf.get_variable('bias', [])

    with tf.name_scope('prediction'):
        unnormalized_prediction = tf.matmul(output_list[-1], output_weight) + bias
        loss = tf.losses.sigmoid_cross_entropy(logits=unnormalized_prediction, multi_class_labels=y_placeholder)
        prediction = tf.sigmoid(unnormalized_prediction)

    return loss, prediction, event_placeholder, context_placeholder, y_placeholder, batch_size, phase_indicator, \
        task_type, time_interval, mutual_intensity, base_intensity


def unit_test(cell_type):
    num_hidden = 16
    context_number = 24
    event_number = 12
    weight_initializer = tf.initializers.orthogonal()
    bias_initializer = tf.initializers.zeros()
    keep_prob = 1.0
    phase_indicator = 2
    num_steps = 10

    if cell_type == 'raw':
        rnn_cell = ContextualRawCell(num_hidden=num_hidden, context_number=context_number, keep_prob=keep_prob,
                                     weight_initializer=weight_initializer, bias_initializer=bias_initializer,
                                     phase_indicator=phase_indicator, event_number=event_number)
    elif cell_type == 'lstm':
        rnn_cell = ContextualLSTMCell(num_hidden=num_hidden, context_number=context_number, keep_prob=keep_prob,
                                      weight_initializer=weight_initializer, bias_initializer=bias_initializer,
                                      phase_indicator=phase_indicator, event_number=event_number)
    elif cell_type == 'gru':
        rnn_cell = ContextualGRUCell(num_hidden=num_hidden, context_number=context_number, keep_prob=keep_prob,
                                     weight_initializer=weight_initializer, bias_initializer=bias_initializer,
                                     phase_indicator=phase_indicator, event_number=event_number)
    else:
        raise ValueError('Wrong Cell Type')

    hawkes_rnn_model(cell=rnn_cell, num_steps=num_steps, num_hidden=num_hidden, num_event=event_number,
                     num_context=context_number, keep_rate_input=keep_prob, event_count=11, auto_encoder_value=15)


if __name__ == '__main__':
    for cell_type_ in ['raw', 'lstm', 'gru']:
        unit_test(cell_type_)
