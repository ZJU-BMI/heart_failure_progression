import tensorflow as tf


def main():
    batch_size = tf.placeholder(tf.int32, [], 'batch_size')
    x_placeholder = tf.placeholder(tf.float32, [10, None, 20], 'x')
    sequence_length = tf.placeholder(tf.float32, [None], 'sequence_length')
    lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=15)
    initial_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
    outputs, state = tf.nn.dynamic_rnn(lstm_cell, x_placeholder,
                                       sequence_length=sequence_length,
                                       initial_state=initial_state,
                                       dtype=tf.float32)


if __name__ == '__main__':
    main()
