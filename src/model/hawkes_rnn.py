# encoding=utf-8-sig
import tensorflow as tf
from rnn_cell import GRUCell
from rnn_cell import LSTMCell
from rnn_cell import RawCell
import autoencoder


def __hawkes_rnn(rnn_cell, input_x, event_list, batch_size, base_intensity,  mutual_intensity, task_type,
                 time_interval, markov_assumption, initializer=tf.initializers.random_normal()):
    """
    :param rnn_cell:
    :param input_x:
    :param event_list:
    :param batch_size:
    :param base_intensity:
    :param mutual_intensity:
    :param time_interval:
    :param task_type:
    :return: loss, prediction, x_placeholder, y_placeholder, batch_size, phase_indicator
    其中 phase_indicator>0代表是测试期，<=0代表是训练期
    """
    # 此处的input_list，指的是可能已经经过DAE降维后的输入
    input_list = tf.unstack(input_x, axis=1)
    event_list = tf.unstack(event_list, axis=1)
    time_interval = tf.unstack(time_interval, axis=1)

    output_list = list()
    recurrent_state = rnn_cell.generate_initial_state(batch_size=batch_size)

    with tf.variable_scope('trans_decay', reuse=tf.AUTO_REUSE):
        trans_decay = tf.get_variable('trans_decay', [1, recurrent_state.shape[1]], initializer=initializer)

    if markov_assumption:
        for i in range(len(input_list)):
            intensity = calculate_intensity_markov(time_interval_list=time_interval, event_list=event_list, index=i,
                                                   base_intensity_vector=base_intensity, task_index=task_type,
                                                   mutual_intensity_matrix=mutual_intensity)
            recurrent_state = recurrent_state * intensity * trans_decay
            output_state, recurrent_state = rnn_cell(input_list[i], recurrent_state)
            output_list.append(output_state)
    else:
        for i in range(len(input_list)):
            intensity = calculate_intensity_full(time_interval_list=time_interval, event_list=event_list, index=i,
                                                 base_intensity_vector=base_intensity, task_index=task_type,
                                                 mutual_intensity_matrix=mutual_intensity)
            # recurrent state的传递损失
            recurrent_state = recurrent_state*intensity*trans_decay
            output_state, recurrent_state = rnn_cell(input_list[i], recurrent_state)
            output_list.append(output_state)

    return output_list


def calculate_intensity_markov(time_interval_list, event_list, base_intensity_vector, mutual_intensity_matrix, index,
                               task_index, omega=-0.006):
    # 如果是第一次入院，感觉intensity怎么取都不太合适，想想还是直接不用了算了
    if index == 0:
        return 0
    # 有关计算互激发，到底是用最终要预测的那个事件，还是本次发生的事件，思考了一会儿，
    # 从拍脑袋决定来看，感觉还是用最终预测的事件比较合理
    else:
        # intensity sum可分为两部分，第一部分是
        intensity_sum = tf.expand_dims(base_intensity_vector[task_index], axis=1)

        time_interval = tf.expand_dims(time_interval_list[index] - time_interval_list[index - 1], axis=1)
        time = tf.exp(time_interval * omega)

        mutual_intensity_vector = tf.expand_dims(mutual_intensity_matrix[task_index], axis=1)
        event = event_list[index-1]
        mutual_intensity = tf.matmul(event, mutual_intensity_vector)
        intensity_sum += mutual_intensity*time
    return intensity_sum


def calculate_intensity_full(time_interval_list, event_list, base_intensity_vector, mutual_intensity_matrix, index,
                             task_index, omega=-0.006):
    # 如果是第一次入院，感觉intensity怎么取都不太合适，想想还是直接不用了算了
    if index == 0:
        return 0
    else:
        intensity_sum = tf.expand_dims(base_intensity_vector[task_index], axis=1)
        for i in range(0, index):
            time_interval = tf.expand_dims(time_interval_list[index] - time_interval_list[i], axis=1)
            time = tf.exp(time_interval * omega)

            mutual_intensity_vector = tf.expand_dims(mutual_intensity_matrix[task_index], axis=1)
            event = event_list[i]
            mutual_intensity = tf.matmul(event, mutual_intensity_vector)
            intensity_sum += mutual_intensity*time
    return intensity_sum


def hawkes_rnn_model(cell, num_steps, num_hidden, num_context, num_event, keep_rate_input, dae_weight,
                     phase_indicator, markov_assumption, auto_encoder_value,
                     auto_encoder_initializer=tf.initializers.orthogonal()):
    """
    :param cell:
    :param num_steps:
    :param num_hidden:
    :param num_context: 要求Context变量全部变为二值变量
    :param num_event:
    :param dae_weight:
    :param keep_rate_input:
    :param markov_assumption:
    :param phase_indicator:
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
            context_placeholder = tf.placeholder(tf.float32, [None, num_steps, num_context], name='context')
            y_placeholder = tf.placeholder(tf.float32, [None, 1], name='y_placeholder')

            task_type = tf.placeholder(tf.int32, shape=[], name="task_type")
            mutual_intensity = tf.placeholder(tf.float32, shape=[num_event, num_event], name="mutual_intensity")
            base_intensity = tf.placeholder(tf.float32, shape=[num_event, 1], name="base_intensity")
            time_interval = tf.placeholder(tf.float32, shape=[None, num_steps], name="time_interval")

            processed_input, input_x, autoencoder_weight = autoencoder.denoising_autoencoder(
                phase_indicator, context_placeholder, event_placeholder, keep_rate_input, auto_encoder_value,
                auto_encoder_initializer)

            output_list = __hawkes_rnn(base_intensity=base_intensity, mutual_intensity=mutual_intensity,
                                       batch_size=batch_size,  task_type=task_type, rnn_cell=cell,
                                       input_x=processed_input, time_interval=time_interval,
                                       markov_assumption=markov_assumption, event_list=event_placeholder)

    with tf.variable_scope('output_layer', reuse=tf.AUTO_REUSE):
        output_weight = tf.get_variable("weight", [num_hidden, 1], initializer=tf.initializers.orthogonal())
        bias = tf.get_variable('bias', [])

    with tf.name_scope('loss'):
        unnormalized_prediction = tf.matmul(output_list[-1], output_weight) + bias
        loss_pred = tf.losses.sigmoid_cross_entropy(logits=unnormalized_prediction, multi_class_labels=y_placeholder)

        if auto_encoder_value > 0:
            loss_dae = autoencoder.autoencoder_loss(embedding=processed_input, origin_input=input_x,
                                                    weight=autoencoder_weight)
        else:
            loss_dae = 0

        loss = loss_pred+loss_dae*dae_weight

    with tf.name_scope('prediction'):
        prediction = tf.sigmoid(unnormalized_prediction)

    return loss, prediction, event_placeholder, context_placeholder, y_placeholder, batch_size, phase_indicator, \
        task_type, time_interval, mutual_intensity, base_intensity


def unit_test(cell_type):
    num_hidden = 16
    embedding_input_length = 12
    weight_initializer = tf.initializers.orthogonal()
    bias_initializer = tf.initializers.zeros()
    keep_prob = 1.0
    num_steps = 10
    num_context = 18
    num_event = 20
    dae_weight = 1

    phase_indicator = tf.placeholder(tf.int32, shape=[], name="phase_indicator")

    if cell_type == 'raw':
        rnn_cell = RawCell(num_hidden=num_hidden, input_length=embedding_input_length, keep_prob=keep_prob,
                           weight_initializer=weight_initializer, bias_initializer=bias_initializer,
                           phase_indicator=phase_indicator)
    elif cell_type == 'lstm':
        rnn_cell = LSTMCell(num_hidden=num_hidden, input_length=embedding_input_length, keep_prob=keep_prob,
                            weight_initializer=weight_initializer, bias_initializer=bias_initializer,
                            phase_indicator=phase_indicator)
    elif cell_type == 'gru':
        rnn_cell = GRUCell(num_hidden=num_hidden, input_length=embedding_input_length, keep_prob=keep_prob,
                           weight_initializer=weight_initializer, bias_initializer=bias_initializer,
                           phase_indicator=phase_indicator)
    else:
        raise ValueError('Wrong Cell Type')

    hawkes_rnn_model(cell=rnn_cell, num_steps=num_steps, num_hidden=num_hidden, num_event=num_event,
                     num_context=num_context, keep_rate_input=keep_prob, markov_assumption=False,
                     auto_encoder_value=embedding_input_length, phase_indicator=phase_indicator, dae_weight=dae_weight)


if __name__ == '__main__':
    for cell_type_ in ['raw', 'lstm', 'gru']:
        tf.reset_default_graph()
        unit_test(cell_type_)
