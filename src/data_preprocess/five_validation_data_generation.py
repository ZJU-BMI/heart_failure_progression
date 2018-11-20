# encoding=utf-8-sig
import csv
import os
import random
import numpy as np
from itertools import islice


def data_filter(feature_data, label_data, data_length):
    # eliminate data with insufficient admission record
    invalid_patient_set = set()
    for patient_id in feature_data:
        # 如果一个RNN的需要三次作为输入，则符合条件的数据至少要入院四次，不然第三次入院产生的标签是没有意义的
        if len(feature_data[patient_id]) <= data_length:
            invalid_patient_set.add(patient_id)
    for patient_id in invalid_patient_set:
        feature_data.pop(patient_id)
        label_data.pop(patient_id)

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

    # get truncated feature sequence and label
    data_dict = dict()
    for patient_id in medium_data_dict:
        feature = list()

        for i in range(data_length):
            feature.append(medium_data_dict[patient_id][i][0])
        label = medium_data_dict[patient_id][data_length - 1][1]
        data_dict[patient_id] = [feature, label]

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


def five_fold_generate(feature_path, label_path, data_length):
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

    fold_size = len(feature_list) // 5
    fold_list = list()
    for i in range(5):
        fold_feature = feature_list[i * fold_size: (i + 1) * fold_size]
        single_fold_label = dict()
        for key in label_dict:
            single_fold_label[key] = label_dict[key][i*fold_size: (i+1)*fold_size]
        fold_list.append([fold_feature, single_fold_label])
    return fold_list


def main(data_length):
    """
    由于本研究的特殊性，数据只做截断不做填充
    例如：设定data_length为5，也就是要求病人有6次入院记录，其中前5次作为序列数据输入RNN，最后一次的Event作为Label
    当一个病人有8次住院记录时，在第六次入院记录时进行数据截断，丢弃第7,8次住院记录。使用前5次入院作为输入
    第六次的Event作为Label（已经完整的完成了标记）
    当一个病人只有4次入院记录时，则弃用这一数据。显然，Data_Length拉的越长，数据越少
    :return:
    """
    feature_path = os.path.abspath('..\\..\\resource\\预处理后的长期纵向数据_特征.csv')
    label_path = os.path.abspath('..\\..\\resource\\预处理后的长期纵向数据_标签.csv')
    save_root = os.path.abspath('..\\..\\resource\\rnn_data\\')

    for j in range(10):
        fold_list = five_fold_generate(feature_path, label_path, data_length)
        for i in range(5):
            feature = fold_list[i][0]
            feature_np = np.array(feature, dtype=np.float64)
            single_feature_path = 'length_{}_repeat_{}_fold_{}_feature.npy'.format(str(data_length), str(j), str(i))
            np.save(os.path.join(save_root, single_feature_path), feature_np)
            label = fold_list[i][1]
            for key in label:
                single_label_path = 'length_{}_repeat_{}_fold_{}_{}_label.npy'.format(str(data_length), str(j),
                                                                                      str(i), key)
                label_np = np.array(label[key], dtype=np.float64)
                np.save(os.path.join(save_root, single_label_path), label_np)


if __name__ == '__main__':
    main(3)
