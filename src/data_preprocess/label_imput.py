# encoding=utf-8-sig
# 根据之前的经验，将心源性标签丢失的数据进行填补
# 训练集是心功能入院且有分级的数据
# 测试集是心功能入院但是分级信息丢失的数据
# 将测试集进行心功能1,2,3,4级，死亡，再血管化分类，然后删除标签丢失的信息
# 根据现有测试，RF能在分级填补上得到大约0.65的ACC
import os
import csv
import numpy as np
import random
from itertools import islice
from sklearn.svm import LinearSVC


def get_data(path):
    train_set = list()
    test_feature = list()

    test_patient_dict = dict()
    test_count = 0
    with open(path, 'r', encoding='gbk', newline='') as file:
        csv_reader = csv.reader(file)
        for line in islice(csv_reader, 2, None):
            # 若一个数据为心源性入院（line[15]）且标签丢失（line[10]），编入测试集；
            # 反之，心源性入院但没有标签丢失的，编入训练（验证）集
            # line[16] 之后是特征， line[5:9]是心功能的四个分级
            # 注意，若之前的数据预处理代码修改导致相应的index意义发生变化，此处要及时修改
            if line[15] == '1' and line[10] == '1':
                test_feature.append(line[16:])
                # 用于记录每个test case对应的patient_id 和 visit_id
                test_patient_dict[test_count] = [line[0], line[1]]
                test_count += 1
                continue
            elif line[15] == '1':
                # 心功能1,2,3,4级，死亡
                event = line[5:9]
                train_set.append([event, line[16:]])
    random.shuffle(train_set)

    # 出于Sklearn的需求，要把心功能1,2,3，4级分级的one hot编码改成有序编码
    # 由于之前的定义策略，可能会出现有心功能分级的入院被更高优先级的事件盖过（手术，死亡）
    # 因此，可能存在一些数据事实上有心功能分级，但是事件标签里没写的情况，这些数据直接跳过
    train_feature = list()
    train_label = list()
    for i in range(len(train_set)):
        for j in range(len(train_set[i][0])):
            if int(train_set[i][0][j]) == 1:
                # 原始的数据设计保证一个i中，只会有一个j==1
                train_feature.append(train_set[i][1])
                train_label.append(j)

    train_feature = np.array(train_feature, dtype=np.float)
    train_label = np.array(train_label, dtype=np.float)
    test_feature = np.array(test_feature, dtype=np.float)
    return train_feature, train_label, test_feature, test_patient_dict


def event_convert(test_prediction, test_patient_dict):
    """
    将预测的结果进行相应的填充
    :param test_prediction:
    :param test_patient_dict:
    :return:
    """
    event_dict = dict()
    one = 0
    two = 0
    three = 0
    four = 0
    for i in range(len(test_prediction)):
        patient_id, visit_id = test_patient_dict[i]
        if not event_dict.__contains__(patient_id):
            event_dict[patient_id] = dict()
        if test_prediction[i] == 0:
            event_dict[patient_id][visit_id] = '心功能1级'
            one += 1
        if test_prediction[i] == 1:
            event_dict[patient_id][visit_id] = '心功能2级'
            two += 1
        if test_prediction[i] == 2:
            event_dict[patient_id][visit_id] = '心功能3级'
            three += 1
        if test_prediction[i] == 3:
            event_dict[patient_id][visit_id] = '心功能3级'
            four += 1
    print('class 1: {}'.format(one))
    print('class 2: {}'.format(two))
    print('class 3: {}'.format(three))
    print('class 4: {}'.format(four))
    return event_dict


def data_regenerate(event_dict, origin_file, write_path):
    """
    读取既有数据，然后按照event_dict的内容填补缺失数据，然后删除数据标签丢失这一项目
    :param event_dict:
    :param write_path:
    :param origin_file:
    :return:
    """
    data_dict = dict()
    feature_dict = dict()
    with open(origin_file, 'r', encoding='gbk', newline='') as file:
        csv_reader = csv.reader(file)
        head_flag = True
        for line in csv_reader:
            if head_flag:
                for i in range(2, len(line)):
                    feature_dict[i] = line[i]
                head_flag = False
                continue

            patient_id = line[0]
            visit_id = line[1]

            if not data_dict.__contains__(patient_id):
                data_dict[patient_id] = dict()
            data_dict[patient_id][visit_id] = dict()

            for i in range(2, len(line)):
                data_dict[patient_id][visit_id][feature_dict[i]] = line[i]

    # 事件替换
    for patient_id in event_dict:
        for visit_id in event_dict[patient_id]:
            event = event_dict[patient_id][visit_id]
            data_dict[patient_id][visit_id][event] = 1
    for patient_id in data_dict:
        for visit_id in data_dict[patient_id]:
            data_dict[patient_id][visit_id].pop('心功能标签丢失')

    # 写数据
    general_list = list()
    for patient_id in data_dict:
        for visit_id in data_dict[patient_id]:
            for key in data_dict[patient_id][visit_id]:
                general_list.append(key)
            break
        break
    data_to_write = list()
    head = list()
    head.append('patient_id')
    head.append('visit_id')
    for item in general_list:
        head.append(item)
    data_to_write.append(head)
    for patient_id in data_dict:
        # 强制有序输出
        for i in range(100):
            if data_dict[patient_id].__contains__(str(i)):
                visit_id = str(i)
                row = [patient_id, visit_id]
                for key in general_list:
                    value = data_dict[patient_id][visit_id][key]
                    row.append(value)
                data_to_write.append(row)
    with open(write_path, 'w', encoding='gbk', newline='') as file:
        csv.writer(file).writerows(data_to_write)

    return data_dict


def main():
    source_data_path = os.path.abspath('..\\..\\resource\\预处理中间结果.csv')
    save_path = os.path.abspath('..\\..\\resource\\预处理后的长期纵向数据_特征.csv')
    train_feature, train_label, test_feature, test_patient_dict = get_data(source_data_path)
    class_weight = {0: 4.1, 1: 1, 2: 2.7, 3: 4}
    clf = LinearSVC(tol=1e-4, max_iter=500, class_weight=class_weight, C=0.1)
    clf.fit(train_feature, train_label)
    test_prediction = clf.predict(test_feature)
    event_dict = event_convert(test_prediction, test_patient_dict)
    data_regenerate(event_dict, source_data_path, save_path)
    print('finish')


if __name__ == '__main__':
    main()
