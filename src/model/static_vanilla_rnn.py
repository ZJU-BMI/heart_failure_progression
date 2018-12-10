# encoding=utf-8-sig
import tensorflow as tf
import autoencoder
from rnn_cell import GRUCell
from rnn_cell import LSTMCell
from rnn_cell import RawCell


def vanilla_rnn_model(cell, num_steps, num_hidden, num_context, num_event, keep_rate_input, dae_weight,
                      phase_indicator, auto_encoder_value, auto_encoder_initializer=tf.initializers.orthogonal()):
    """
    :param cell:
    :param num_steps:
    :param num_hidden:
    :param num_context: 要求Context变量全部变为二值变量
    :param num_event:
    :param dae_weight:
    :param keep_rate_input:
    :param phase_indicator:
    :param auto_encoder_value: 大于0时执行对输入的自编码，值即为最终降到的维度
    :param auto_encoder_initializer:
    :return:
    loss, prediction, x_placeholder, y_placeholder, batch_size, phase_indicator
    其中 phase_indicator>0代表是测试期，<=0代表是训练期
    """
    with tf.name_scope('vanilla_rnn'):
        with tf.name_scope('data_source'):
            batch_size = tf.placeholder(tf.int32, [], name='batch_size')
            # 标准输入规定为BTD
            event_placeholder = tf.placeholder(tf.float32, [None, num_steps, num_event], name='event_placeholder')
            context_placeholder = tf.placeholder(tf.float32, [None, num_steps, num_context], name='context_placeholder')
            y_placeholder = tf.placeholder(tf.float32, [None, 1], name='y_placeholder')

            # input_x 用于计算重构原始向量时产生的误差
            processed_input, input_x, autoencoder_weight = autoencoder.denoising_autoencoder(
                phase_indicator, context_placeholder, event_placeholder, keep_rate_input, auto_encoder_value,
                auto_encoder_initializer)

            output_list = __vanilla_rnn(batch_size=batch_size,  rnn_cell=cell, input_x=processed_input)

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

    return loss, prediction, event_placeholder, context_placeholder, y_placeholder, batch_size, phase_indicator


def __vanilla_rnn(rnn_cell, input_x, batch_size):
    """
    :param rnn_cell:
    :param input_x:
    :param batch_size:
    :return: loss, prediction, x_placeholder, y_placeholder, batch_size, phase_indicator
    其中 phase_indicator>0代表是测试期，<=0代表是训练期
    """
    input_list = tf.unstack(input_x, axis=1)

    output_list = list()
    recurrent_state = rnn_cell.generate_initial_state(batch_size=batch_size)

    for i in range(len(input_list)):
        output_state, recurrent_state = rnn_cell(input_list[i], recurrent_state)
        output_list.append(output_state)

    return output_list


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

    vanilla_rnn_model(cell=rnn_cell, num_steps=num_steps, num_hidden=num_hidden, num_event=num_event,
                      num_context=num_context, keep_rate_input=keep_prob, auto_encoder_value=embedding_input_length,
                      phase_indicator=phase_indicator, dae_weight=dae_weight)


if __name__ == '__main__':
    for cell_type_ in ['raw', 'lstm', 'gru']:
        tf.reset_default_graph()
        unit_test(cell_type_)
