# encoding=utf-8-sig
# concat hawkes和fuse hawkes（也就是dynamic rnn）的主要区别在于，后者直接把输入的event和context拼接输入一个RNN
# 前者把event和context分别输入两个RNN。然后分别拼接隐藏层。
# 中间的HP Node操作过程没有区别。因此，Concat Hawkes基本可以复用dynamic rnn完成
import numpy as np
import tensorflow as tf
from rnn_cell import GRUCell, LSTMCell, RawCell
import autoencoder
from dynamic_hawkes_rnn import _hawkes_dynamic_rnn


def concat_hawkes_model(cell_context, cell_event, num_steps, num_hidden, num_context, num_event, keep_rate_input,
                        dae_weight, phase_indicator, autoencoder_length,
                        autoencoder_initializer=tf.initializers.orthogonal()):
    with tf.name_scope('data_source'):
        # 标准输入规定为TBD
        batch_size = tf.placeholder(tf.int32, [], name='batch_size')
        event_placeholder = tf.placeholder(tf.float32, [num_steps, None, num_event], name='event_placeholder')
        context_placeholder = tf.placeholder(tf.float32, [num_steps, None, num_context], name='context_placeholder')
        base_intensity = tf.placeholder(tf.float32, [num_event, 1], name='base_intensity')
        mutual_intensity = tf.placeholder(tf.float32, [num_event, num_event], name='mutual_intensity')
        time_list = tf.placeholder(tf.int32, [None, num_steps], name='time_list')
        task_index = tf.placeholder(tf.int32, [], name='task_index')
        sequence_length = tf.placeholder(tf.int32, [None], name='sequence_length')
        y_placeholder = tf.placeholder(tf.float32, [None, 1], name='y_placeholder')

        event_initial_state = cell_event.get_initial_state(batch_size)
        context_initial_state = cell_context.get_initial_state(batch_size)

    with tf.name_scope('autoencoder'):
        # input_x 用于计算重构原始向量时产生的误差
        processed_input, autoencoder_weight = autoencoder.denoising_autoencoder(
            phase_indicator, context_placeholder, keep_rate_input, autoencoder_length, autoencoder_initializer)

    with tf.name_scope('context_hawkes_rnn'):
        output_final, final_state = _hawkes_dynamic_rnn(cell_context, processed_input, sequence_length,
                                                        context_initial_state, base_intensity=base_intensity,
                                                        task_index=task_index, mutual_intensity=mutual_intensity,
                                                        time_list=time_list, event_list=event_placeholder,
                                                        scope='context_hawkes')
        output_length = output_final.shape[2].value
        state_length = final_state.shape[1].value
        if output_length == state_length:
            # 不需要做任何事情
            context_final_state = final_state
        elif output_length * 2 == state_length:
            context_final_state = tf.split(final_state, 2, axis=1)[0]
        else:
            raise ValueError('Invalid Size')

    with tf.name_scope('event_hawkes_rnn'):
        output_final, final_state = _hawkes_dynamic_rnn(cell_event, event_placeholder, sequence_length,
                                                        event_initial_state, base_intensity=base_intensity,
                                                        task_index=task_index, mutual_intensity=mutual_intensity,
                                                        time_list=time_list, event_list=event_placeholder,
                                                        scope='event_hawkes')
        output_length = output_final.shape[2].value
        state_length = final_state.shape[1].value
        if output_length == state_length:
            # 不需要做任何事情
            event_final_state = final_state
        elif output_length * 2 == state_length:
            event_final_state = tf.split(final_state, 2, axis=1)[0]
        else:
            raise ValueError('Invalid Size')

    with tf.name_scope('state_concat'):
        concat_final_state = tf.concat([event_final_state, context_final_state], axis=1)

    with tf.variable_scope('output_para'):
        output_weight = tf.get_variable("weight", [num_hidden*2, 1], initializer=tf.initializers.orthogonal())
        bias = tf.get_variable('bias', [])

    with tf.name_scope('prediction'):
        unnormalized_prediction = tf.matmul(concat_final_state, output_weight) + bias
        prediction = tf.sigmoid(unnormalized_prediction)

    with tf.name_scope('loss'):
        with tf.name_scope('pred_loss'):
            loss_pred = tf.losses.sigmoid_cross_entropy(logits=unnormalized_prediction, multi_class_labels=y_placeholder)

        with tf.name_scope('dae_loss'):
            if autoencoder_length > 0:
                loss_dae = autoencoder.autoencoder_loss(embedding=processed_input, origin_input=context_placeholder,
                                                        weight=autoencoder_weight)
            else:
                loss_dae = 0

        with tf.name_scope('loss_sum'):
            loss = loss_pred + loss_dae * dae_weight

    return loss, prediction, event_placeholder, context_placeholder, y_placeholder, batch_size, phase_indicator, \
        base_intensity, mutual_intensity, time_list, task_index, sequence_length


def unit_test():
    num_hidden = 10
    num_steps = 20
    keep_prob = 1.0
    num_context = 50
    num_event = 30
    keep_rate_input = 0.8
    dae_weight = 1
    autoencoder_length = 15
    if autoencoder_length > 0:
        context_input_length = autoencoder_length
    else:
        context_input_length = num_context
    event_input_length = num_event

    initializer_o = tf.initializers.orthogonal()
    initializer_z = tf.initializers.zeros()
    phase_indicator = tf.placeholder(tf.int16, [])

    # 试验阶段
    test_cell_type = 1
    if test_cell_type == 0:
        a_cell_event = GRUCell(num_hidden=num_hidden, input_length=event_input_length, weight_initializer=initializer_o,
                               bias_initializer=initializer_z, keep_prob=keep_prob, phase_indicator=phase_indicator,
                               name='event_gru')
        a_cell_context = GRUCell(num_hidden=num_hidden, input_length=context_input_length, keep_prob=keep_prob,
                                 bias_initializer=initializer_z,  phase_indicator=phase_indicator, name='context_gru',
                                 weight_initializer=initializer_o)
        loss, prediction, event_placeholder, context_placeholder, y_placeholder, batch_size, phase_indicator, \
            base_intensity, mutual_intensity, time_list, task_index, sequence_length = \
            concat_hawkes_model(a_cell_context, a_cell_event, num_steps, num_hidden, num_context, num_event,
                                keep_rate_input, dae_weight, phase_indicator, autoencoder_length)
    elif test_cell_type == 1:
        b_cell_event = RawCell(num_hidden=num_hidden, weight_initializer=initializer_o, bias_initializer=initializer_z,
                               keep_prob=keep_prob, input_length=event_input_length, phase_indicator=phase_indicator,
                               name='event_raw')
        b_cell_context = RawCell(num_hidden=num_hidden, weight_initializer=initializer_o, keep_prob=keep_prob,
                                 bias_initializer=initializer_z,  input_length=context_input_length,
                                 phase_indicator=phase_indicator, name='context_raw')
        loss, prediction, event_placeholder, context_placeholder, y_placeholder, batch_size, phase_indicator, \
            base_intensity, mutual_intensity, time_list, task_index, sequence_length = \
            concat_hawkes_model(b_cell_context, b_cell_event, num_steps, num_hidden, num_context, num_event,
                                keep_rate_input, dae_weight, phase_indicator, autoencoder_length)

    elif test_cell_type == 2:
        c_cell_event = LSTMCell(num_hidden=num_hidden, input_length=event_input_length, name='event_lstm',
                                bias_initializer=initializer_z, keep_prob=keep_prob, phase_indicator=phase_indicator,
                                weight_initializer=initializer_o)
        c_cell_context = LSTMCell(num_hidden=num_hidden, input_length=context_input_length, name='context_lstm',
                                  bias_initializer=initializer_z, keep_prob=keep_prob, phase_indicator=phase_indicator,
                                  weight_initializer=initializer_o)
        loss, prediction, event_placeholder, context_placeholder, y_placeholder, batch_size, phase_indicator, \
            base_intensity, mutual_intensity, time_list, task_index, sequence_length = \
            concat_hawkes_model(c_cell_context, c_cell_event, num_steps, num_hidden, num_context, num_event,
                                keep_rate_input, dae_weight, phase_indicator, autoencoder_length)
    else:
        raise ValueError('')

    batch_size_value = 32
    event = np.random.normal(0, 1, [num_steps, batch_size_value, num_event])
    context_ = np.random.normal(0, 1, [num_steps, batch_size_value, num_context])
    base_intensity_value = np.random.uniform(0, 1, [num_event, 1])
    mutual_intensity_value = np.random.uniform(0, 1, [num_event, num_event])
    time_list_value = np.random.uniform(0, 1, [batch_size_value, num_steps])

    sequence_length_value = np.random.randint(1, 8, [batch_size_value])
    feed_dict = {event_placeholder: event, context_placeholder: context_, batch_size: batch_size_value,
                 phase_indicator: 1, sequence_length: sequence_length_value, base_intensity: base_intensity_value,
                 mutual_intensity: mutual_intensity_value, time_list: time_list_value, task_index: 0}

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        pred = sess.run(prediction, feed_dict=feed_dict)
        print(pred)


if __name__ == '__main__':
    unit_test()


