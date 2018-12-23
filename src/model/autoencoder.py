import tensorflow as tf
# 注意，本实验要求Autoencoder所编码的所有数据都是二值数据，不然非线性函数会非常难取


def denoising_autoencoder(phase_indicator, input_placeholder, keep_rate_input, embedded_size, auto_encoder_initializer):
    """
    输入均为TBD格式
    :param phase_indicator:
    :param input_placeholder:
    :param keep_rate_input:
    :param embedded_size:
    :param auto_encoder_initializer:
    :return:
    """
    length_input = input_placeholder.shape[2]
    with tf.name_scope('add_noise'):
        # 比较准确的来说，我们在这里实现的DAE和原版DAE的设计策略不太一样
        # 我们的这种设计理论上来讲，比较准确的说法是：带dropout的autoencoder
        # dropout的主要特点是，在丢弃数据的同时，对留下的数据放大，使得处理后的数据期望和原始数据一致（这个做法靠不靠谱另说）
        # 但原版的DAE其实不是这么做的，原版的DAE在对输入进行随机丢弃的时候，并不会对留下来的数据进行任何处理
        # 不过在这里我觉得还是带dropout的autoencoder更为靠谱一些，并且用了DAE的名字，特此说明一下
        # 此处的phase indicator用于区分训练期和测试期
        input_dropout = tf.cond(phase_indicator > 0,
                                lambda: input_placeholder,
                                lambda: tf.nn.dropout(input_placeholder, keep_rate_input))

    with tf.name_scope('ae'):
        # 上面的步骤可以视为已经加入了噪声，因此此处只需要降维即可
        if embedded_size > 0:
            with tf.variable_scope('auto_encoder_parameter'):
                autoencoder_weight = tf.get_variable('weight', [length_input, embedded_size],
                                                     initializer=auto_encoder_initializer)

            unstacked_list = tf.unstack(input_dropout, axis=0)
            coded_list = list()
            for single_input in unstacked_list:
                coded_list.append(tf.sigmoid(tf.matmul(single_input, autoencoder_weight)))
            # 确保输入格式为 BTD
            processed_input = tf.convert_to_tensor(coded_list)

        else:
            processed_input = input_dropout
            autoencoder_weight = 0
    return processed_input, autoencoder_weight


def autoencoder_loss(origin_input, embedding, weight):
    weight_tied = tf.transpose(weight)
    embedding_unstack = tf.unstack(embedding, axis=0)

    reconstruct_list = list()
    for i in range(len(embedding_unstack)):
        reconstructed = tf.matmul(embedding_unstack[i], weight_tied)
        reconstruct_list.append(reconstructed)
    reconstruct_input = tf.convert_to_tensor(reconstruct_list)

    loss = tf.losses.sigmoid_cross_entropy(logits=reconstruct_input, multi_class_labels=origin_input)
    return loss


def unit_test():
    num_steps = 10
    batch_size = None
    embedding_size = 20
    context_num = 25
    keep_rate = 1.0
    init = tf.initializers.random_normal()

    phase_indicator = tf.placeholder(tf.int16, [])
    context = tf.placeholder(tf.float32, [num_steps, batch_size, context_num])
    denoising_autoencoder(phase_indicator, context, keep_rate,
                          embedding_size, init)


if __name__ == '__main__':
    unit_test()
