# encoding=utf-8-sig
import csv
import os
import random
import numpy as np
from itertools import islice


def data_filter(feature_data, label_data, max_data_length):
    # sort feature and label
    medium_data_dict = dict()
    for patient_id in feature_data:
        if not medium_data_dict.__contains__(patient_id):
            medium_data_dict[patient_id] = list()
        for i in range(100):
            visit_id = str(i)
            feature_dict = feature_data[patient_id]
            if feature_dict.__contains__(visit_id):
                data = [feature_data[patient_id][visit_id], label_data[patient_id][visit_id]]
                medium_data_dict[patient_id].append(data)

    # 数据的截断与补齐
    data_dict = dict()
    for patient_id in medium_data_dict:
        feature = list()
        feature_num = len(medium_data_dict[patient_id][0][0])

        # 最后一次输入其实是作为标签用的，其特征事实上不发挥作用，因此有效长度是总长度-1
        valid_length = len(medium_data_dict[patient_id])-1

        # 如果有效长度超过了最大长度，则截断，反之补零
        if valid_length > max_data_length:
            valid_length = max_data_length
        for i in range(max_data_length):
            if i < valid_length:
                feature.append(medium_data_dict[patient_id][i][0])
            else:
                feature.append(['0' for _ in range(feature_num)])

        # 最后一次有效输入所对应的标签事实上是真实标签,
        label = medium_data_dict[patient_id][valid_length-1][1]
        feature = np.array(feature, dtype=np.int16)
        data_dict[patient_id] = [feature, label, valid_length]

    data_list = list()
    for patient_id in data_dict:
        data_list.append(data_dict[patient_id])

    return data_list


def read_feature(feature_path):
    feature_dict = dict()
    feature_path = feature_path
    with open(feature_path, 'r', encoding='gbk', newline='') as file:
        csv_reader = csv.reader(file)
        for line in islice(csv_reader, 1, None):
            patient_id, visit_id = line[0], line[1]
            feature = line[2:]
            if not feature_dict.__contains__(patient_id):
                feature_dict[patient_id] = dict()
            feature_dict[patient_id][visit_id] = feature

            # 本语句只是为了检查visit_ID是否可以强制类型转换，在此处无意义
            float(visit_id)
    return feature_dict


def read_label(label_path):
    label_dict = dict()
    with open(label_path, 'r', encoding='gbk', newline='') as file:
        csv_reader = csv.reader(file)
        head_dict = dict()
        for i, line in enumerate(csv_reader):
            if i == 0:
                for j in range(2, len(line)):
                    head_dict[j] = line[j]
                continue

            patient_id, visit_id = line[0], line[1]
            if not label_dict.__contains__(patient_id):
                label_dict[patient_id] = dict()
            label_dict[patient_id][visit_id] = dict()
            for j in range(2, len(line)):
                label_dict[patient_id][visit_id][head_dict[j]] = line[j]
    return label_dict


def six_fold_generate(feature_path, label_path, data_length):
    label_dict = read_label(label_path)
    feature_dict = read_feature(feature_path)
    raw_data = data_filter(feature_dict, label_dict, data_length)
    # 数据随机化
    random.shuffle(raw_data)

    # 抽取特征矩阵
    feature_list = list()
    for item in raw_data:
        feature_list.append(item[0])

    # 抽取标签矩阵
    label_dict = dict()
    for item in raw_data:
        for key in item[1]:
            if not label_dict.__contains__(key):
                label_dict[key] = list()
            label_dict[key].append(item[1][key])

    sequence_length_list = list()
    for item in raw_data:
        sequence_length_list.append(item[2])

    fold_size = len(feature_list) // 6
    fold_list = list()
    for i in range(6):
        fold_feature = feature_list[i * fold_size: (i + 1) * fold_size]
        fold_length = sequence_length_list[i * fold_size: (i + 1) * fold_size]
        single_fold_label = dict()
        for key in label_dict:
            single_fold_label[key] = label_dict[key][i*fold_size: (i+1)*fold_size]
        fold_list.append([fold_feature, single_fold_label, fold_length])
    return fold_list


def main(data_length):
    """
    设定Data Length为n
    则病人至少要有n+1次的有效住院记录，其中
    1. 预处理后的长期纵向数据_特征 中取n个数据
    2. 预处理后的长期纵向数据_标签 中取第n个数据作为标签（这个标签事实上是第n+1次入院的信息赋予的）

    当病人的入院次数大于n+1时，数据在n处截断
    当病人的入院次数小于n+1，比如病人的总入院次数是n-2
    则，取病人的前n-3次入院数据作为feature，剩余部分补零。第n-3次入院对应的label为标签
    :return:
    """
    feature_path = os.path.abspath('../../resource/预处理后的长期纵向数据_特征.csv')
    label_path = os.path.abspath('../../resource/preprocessed_label.csv')
    save_root = os.path.abspath('../../resource/rnn_data/')

    for j in range(10):
        fold_list = six_fold_generate(feature_path, label_path, data_length)
        for i in range(6):
            if i != 5:
                feature = fold_list[i][0]
                feature_np = np.array(feature, dtype=np.float64)
                single_feature_path = 'length_{}_repeat_{}_fold_{}_feature.npy'.format(str(data_length), str(j), str(i))
                np.save(os.path.join(save_root, single_feature_path), feature_np)

                sequence_length = fold_list[i][2]
                sequence_length = np.array(sequence_length, dtype=np.int32)
                sequence_length_path = \
                    'length_{}_repeat_{}_fold_{}_sequence_length.npy'.format(str(data_length), str(j), str(i))
                np.save(os.path.join(save_root, sequence_length_path), sequence_length)
                label = fold_list[i][1]
                for key in label:
                    single_label_path = 'length_{}_repeat_{}_fold_{}_{}_label.npy'.format(str(data_length), str(j),
                                                                                          str(i), key)
                    label_np = np.array(label[key], dtype=np.float64)
                    np.save(os.path.join(save_root, single_label_path), label_np)
            else:
                feature = fold_list[i][0]
                feature_np = np.array(feature, dtype=np.float64)
                single_feature_path = 'length_{}_test_feature.npy'.format(str(data_length))
                np.save(os.path.join(save_root, single_feature_path), feature_np)

                sequence_length = fold_list[i][2]
                sequence_length = np.array(sequence_length, dtype=np.int32)
                sequence_length_path = \
                    'length_{}_test_sequence_length.npy'.format(str(data_length), str(j), str(i))
                np.save(os.path.join(save_root, sequence_length_path), sequence_length)

                label = fold_list[i][1]
                for key in label:
                    single_label_path = 'length_{}_test_{}_label.npy'.format(str(data_length), key)
                    label_np = np.array(label[key], dtype=np.float64)
                    np.save(os.path.join(save_root, single_label_path), label_np)


if __name__ == '__main__':
    for i in range(3, 10):
        main(i)

