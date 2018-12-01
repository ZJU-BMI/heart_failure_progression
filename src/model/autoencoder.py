import tensorflow as tf
# 注意，本实验要求Autoencoder所编码的所有数据都是二值数据，不然非线性函数会非常难取


def denoising_autoencoder(phase_indicator, context_placeholder, event_placeholder, keep_rate_input,
                          autoencoder_value, auto_encoder_initializer):

    num_context = context_placeholder.shape[2]
    num_event = event_placeholder.shape[2]
    with tf.name_scope('dropout'):
        # 用于判断是训练阶段还是测试阶段，用于判断是否要加Dropout
        # 此处的Dropout可以视为加入了Nosing
        # 为确保信息不丢失，dropout只加在context信息上
        context_dropout = tf.cond(phase_indicator > 0,
                                  lambda: context_placeholder,
                                  lambda: tf.nn.dropout(context_placeholder, keep_rate_input))

    with tf.name_scope('dae'):
        # 上面的步骤可以视为已经加入了噪声，因此此处只需要降维即可
        input_x = tf.concat([context_dropout, event_placeholder], axis=2)
        if autoencoder_value > 0:
            with tf.variable_scope('auto_encoder', reuse=tf.AUTO_REUSE):
                autoencoder_weight = tf.get_variable('autoencoder', [num_context + num_event, autoencoder_value],
                                                     initializer=auto_encoder_initializer)

            unstacked_list = tf.unstack(input_x, axis=1)
            coded_list = list()
            for single_input in unstacked_list:
                coded_list.append(tf.sigmoid(tf.transpose(tf.matmul(single_input, autoencoder_weight))))
            # 确保输入格式为 BTD
            processed_input = tf.transpose(tf.convert_to_tensor(coded_list), [2, 0, 1])

        else:
            processed_input = input_x
            autoencoder_weight = 0
    return processed_input, input_x, autoencoder_weight


def autoencoder_loss(origin_input, embedding, weight):
    weight_tied = tf.transpose(weight)
    embedding_unstack = tf.unstack(embedding, axis=1)

    reconstruct_list = list()
    for i in range(len(embedding_unstack)):
        reconstructed = tf.matmul(embedding_unstack[i], weight_tied)
        reconstruct_list.append(reconstructed)
    reconstruct_input = tf.transpose(tf.convert_to_tensor(reconstruct_list), [1, 0, 2])

    loss = tf.losses.sigmoid_cross_entropy(logits=reconstruct_input, multi_class_labels=origin_input)
    return loss


