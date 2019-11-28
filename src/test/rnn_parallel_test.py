# encoding=utf-8-sig
import tensorflow as tf
import numpy as np
import datetime

sequence_length = 500
element_size = 300
batch_size = 100
parallel_iterations = 100
time_major = True
cell = tf.nn.rnn_cell.LSTMCell(128)
inputs = tf.placeholder(tf.float32, [sequence_length, batch_size, element_size])
outputs, state = tf.nn.dynamic_rnn(cell, inputs, time_major=True,
                                   parallel_iterations=parallel_iterations, dtype=tf.float32)
init_node = tf.global_variables_initializer()

input_value = np.random.normal(0, 1, [sequence_length, batch_size, element_size])

with tf.Session() as sess:
    sess.run(init_node)
    start_time = datetime.datetime.now()
    for i in range(400):
        sess.run(state, feed_dict={inputs: input_value})
    end_time = datetime.datetime.now()
    time_cost = (end_time-start_time).seconds
    print('time cost {}, para = {}'.format(time_cost, parallel_iterations))
