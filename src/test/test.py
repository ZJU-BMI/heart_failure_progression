# encoding=utf-8-sig
import tensorflow as tf
import numpy as np

x = tf.placeholder(tf.float32, [None, 10, 20])
lstm = tf.nn.rnn_cell.LSTMCell(5)
rnn = tf.keras.layers.RNN(lstm)
output_list = rnn(x)

init = tf.global_variables_initializer()
print('finish')

with tf.Session() as sess:
    sess.run(init)
    output_list = sess.run(output_list, feed_dict={x: np.random.normal(0, 1, [2, 10, 20])})
