# encoding=utf-8-sig
# 基于Hawkes进行Attention机制的策略分析
# 基于Hawkes过程所选取的数据直接指导RNN的关注权重，到时候要和Vanilla RNN比较，此处的权重不是学的，而是来源于外源性指导
# Hawkes Attention的机制是，Base Intensity乘以一个（要学习的）系数，作为事件的基准概率；然后以互激发强度做Attention
# 直接作为各个Hidden State的权重，然后取平均，乘矩阵算一个概率，和基准概率相加，作为最终判断概率
import tensorflow as tf
import os
from rnn_dropout import drop_rnn
import csv


def hawkes_attention(num_steps, num_hidden, num_feature, keep_rate, mutual_intensity_matrix, base_intensity_vector,
                     event_count, omega=0.006):
    """
    :param num_steps:
    :param num_hidden:
    :param num_feature:
    :param keep_rate:
    :param mutual_intensity_matrix: numpy 矩阵 Event*Event， e+_ij元素指代j激发i的强度
    :param base_intensity_vector: numpy矩阵
    :param event_count: 独立事件的个数
    :param omega:
    :return:
    """
    batch_size = tf.placeholder(tf.int32, [], name='batch_size')
    x_placeholder = tf.placeholder(tf.float32, [None, num_steps, num_feature], name='x_placeholder')
    y_placeholder = tf.placeholder(tf.float32, [None, 1], name='y_placeholder')
    phase_indicator = tf.placeholder(tf.int32, shape=[], name="phase_indicator")
    # 半年取90， 一年取180， 两年取365
    task_time = tf.placeholder(tf.float32, shape=[], name="task_time")
    # 按照一开始的Event标定序列，直接输入相应数字
    task_type = tf.placeholder(tf.int32, shape=[], name="task_type")

    output_list = drop_rnn(num_steps, num_hidden, num_feature, keep_rate, x_placeholder, phase_indicator, batch_size)

    with tf.name_scope('get_intensity'):
        mutual_intensity_tensor = tf.convert_to_tensor(mutual_intensity_matrix, dtype=tf.float32)
        base_intensity_tensor = tf.convert_to_tensor(base_intensity_vector, dtype=tf.float32)
        output_list = tf.convert_to_tensor(output_list)

        # 从原始数据中获取Event和时间信息，注意，此处要求Event一定要在输入数据的最前面
        # 要求时间一定恰好跟在Event信息后面，如果后期数据语义发生改变，则会造成问题
        event_list = tf.slice(x_placeholder, [0, 0, 0], [-1, -1, event_count])
        time_list = tf.slice(x_placeholder, [0, 0, event_count], [-1, -1, 1])

        # 获取各个事件距离目标时间窗口的时间差，并计算出相应的时间激发强度
        time_max = tf.reduce_max(time_list)
        time_interval_list = task_time + time_max - time_list
        time_intensity = tf.exp(-omega*time_interval_list)

        # 获取基准线强度
        base_intensity = base_intensity_tensor[task_type]

        # 获取互激发矩阵强度
        # 互激发矩阵中的元素e_ij指代第i类事件激发第j类事件的强度
        mutual_intensity_vector = tf.expand_dims(mutual_intensity_tensor[task_type, :], axis=1)
        mutual_intensity_weight = tf.map_fn(lambda x: tf.matmul(x, mutual_intensity_vector), event_list)

        # 基于上述强度指导三个权重的分配
        mutual_intensity_weight = time_intensity*mutual_intensity_weight
        mutual_sum = tf.reduce_sum(mutual_intensity_weight, axis=1, keepdims=True)
        mutual_intensity_weight = mutual_intensity_weight / mutual_sum

    with tf.variable_scope('output_weight'):
        output_weight = tf.get_variable("output_weight", [num_hidden, 1], initializer=tf.initializers.orthogonal())
        base = tf.get_variable('base', [])
        bias = tf.get_variable('bias', [])

    with tf.name_scope('attention_and_output'):
        output_list = tf.transpose(output_list, [1, 0, 2])
        fusion_state = tf.reduce_sum(mutual_intensity_weight*output_list, axis=1)
        unnormalized_prediction = tf.matmul(fusion_state, output_weight) + bias + base*base_intensity
        loss = tf.losses.sigmoid_cross_entropy(logits=unnormalized_prediction, multi_class_labels=y_placeholder)

        prediction = tf.sigmoid(unnormalized_prediction)

    return loss, prediction, x_placeholder, y_placeholder, batch_size, phase_indicator, task_time, task_type


def read_mutual_intensity(file_path):
    mutual_intensity_dict = dict()
    head_dict = dict()
    with open(file_path, 'r', encoding='gbk', newline='') as file:
        csv_reader = csv.reader(file)
        row = 0
        for line in csv_reader:
            if row == 0:
                for i in range(1, len(line)):
                    head_dict[i] = line[i]
                    mutual_intensity_dict[line[i]] = dict()
            else:
                current_item = None
                for i in range(len(line)):
                    if i == 0:
                        current_item = line[0]
                    else:
                        mutual_intensity_dict[head_dict[i]][current_item] = line[i]
            row += 1
    return mutual_intensity_dict


def read_base_intensity(file_path):
    base_intensity_dict = dict()
    head_dict = dict()
    with open(file_path, 'r', encoding='gbk', newline='') as file:
        csv_reader = csv.reader(file)
        row = 0
        for line in csv_reader:
            if row == 0:
                for i in range(len(line)):
                    head_dict[i] = line[i]
            if row == 1:
                for i in range(len(line)):
                    base_intensity_dict[head_dict[i]] = float(line[i])
            row += 1
    return base_intensity_dict


if __name__ == '__main__':
    base_intensity_path = os.path.abspath('..\\..\\resource\\hawkes_result\\base.csv')
    mutual_intensity_path = os.path.abspath('..\\..\\resource\\hawkes_result\\mutual.csv')
    read_base_intensity(base_intensity_path)
    read_mutual_intensity(mutual_intensity_path)
