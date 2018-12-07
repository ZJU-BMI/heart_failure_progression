# encoding=utf-8-sig
import tensorflow as tf
import numpy as np
import datetime

parallel_iteration = 1
max_iteration = 1000
dim_num = 2000
time_index = tf.constant(0, dtype=tf.int32)
output_ = tf.constant(1, dtype=tf.float32, shape=[dim_num, dim_num])

place_list = list()
for i in range(10):
    place_list.append(tf.placeholder(tf.float32, [dim_num, dim_num]))


def condition(x, _):
    return x < tf.constant(max_iteration)


def body(x, *placeholder_list):
    return_list = list()
    for item_ in placeholder_list:
        cal = item_
        cal = 2 * tf.matmul(cal, cal)/tf.matmul(cal, cal)
        return_list.append(cal)
    return x+1, return_list


_, output_matrix, output_product = tf.while_loop(cond=condition,
                                                 body=body,
                                                 loop_vars=[time_index, place_list],
                                                 parallel_iterations=parallel_iteration)
feed_dict = dict()
for item in place_list:
    feed_dict[item] = np.random.normal(0, 1, [dim_num, dim_num])

with tf.Session() as sess:
    start_time = datetime.datetime.now()
    sess.run(output_matrix, feed_dict=feed_dict)
    end_time = datetime.datetime.now()
    time_cost = (end_time-start_time).seconds
    print('time cost {}, para = {}'.format(time_cost, parallel_iteration))
