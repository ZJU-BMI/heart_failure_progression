# encoding=utf-8
import csv
import os
import copy
import numpy as np
from sklearn.linear_model import LogisticRegression


def read_data(file_path):
    data_dict = dict()
    feature_dict = dict()
    with open(file_path, 'r', encoding='gbk', newline='') as file:
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
    return data_dict


def missing_rate_calc(data_dict):
    missing_dict = dict()
    # 建立字典
    for patient_id in data_dict:
        for visit_id in data_dict[patient_id]:
            for feature in data_dict[patient_id][visit_id]:
                missing_dict[feature] = 0
            break
        break
    row_count = 0
    for patient_id in data_dict:
        for visit_id in data_dict[patient_id]:
            row_count += 1
            for feature in data_dict[patient_id][visit_id]:
                value = data_dict[patient_id][visit_id][feature]
                if feature.__contains__('lost') and value == '1':
                    missing_dict[feature] += 1
    for feature in missing_dict:
        missing_dict[feature] = missing_dict[feature] / row_count
    return missing_dict


def feature_discard(origin_data, feature_missing_rate, white_list, feature_threshold=0.3):
    """
    限于数据特点，必须要有一些可以豁免的白名单存在
    :param origin_data:
    :param feature_missing_rate:
    :param white_list:
    :param feature_threshold:
    :return:
    """
    discard_item_set = set()
    # 丢弃缺失值超过上限的数据
    for patient_id in origin_data:
        for visit_id in origin_data[patient_id]:
            for item in feature_missing_rate:
                missing_rate = float(feature_missing_rate[item])
                if missing_rate > feature_threshold and not white_list.__contains__(item):
                    actual_feature = item.split('_')[0]
                    for item_name in origin_data[patient_id][visit_id]:
                        if item_name.__contains__(actual_feature):
                            discard_item_set.add(item_name)
            break
        break
    for patient_id in origin_data:
        for visit_id in origin_data[patient_id]:
            for item_name in discard_item_set:
                origin_data[patient_id][visit_id].pop(item_name)
    return origin_data


def feature_imputation(white_list_imputed_data):
    """
    由于缺失数据实在比较分散，策略如下
    1. 在预测目标特征A的缺失值时，将剩余所有特征中的缺失值以dummy variable的形式编码，作为特征编入数据
    2. 在预测完毕后，把所有的lost dummy variable删除
    :param white_list_imputed_data:
    :return:
    """
    target_data = copy.deepcopy(white_list_imputed_data)

    # 构建预测列表
    estimate_feature_list = list()
    for patient_id in white_list_imputed_data:
        for visit_id in white_list_imputed_data[patient_id]:
            for feature in white_list_imputed_data[patient_id][visit_id]:
                if feature.__contains__('lost'):
                    estimate_feature_list.append(feature.split('_')[0])
            break
        break

    for item in estimate_feature_list:
        # 构建训练集与测试集，以及对应关系
        mapping_dict = dict()
        train_feature = list()
        train_label = list()
        test_feature = list()

        # 构建训练集与测试集
        for patient_id in white_list_imputed_data:
            for visit_id in white_list_imputed_data[patient_id]:
                feature_tuple = list()
                if white_list_imputed_data[patient_id][visit_id][item+'_lost'] == '1':
                    for feature in white_list_imputed_data[patient_id][visit_id]:
                        if feature.__contains__(item):
                            continue
                        else:
                            value = white_list_imputed_data[patient_id][visit_id][feature]
                            feature_tuple.append(value)
                    test_feature.append(feature_tuple)
                    mapping_dict[len(test_feature)-1] = [patient_id, visit_id]
                else:
                    for feature in white_list_imputed_data[patient_id][visit_id]:
                        if feature.__contains__(item):
                            value = white_list_imputed_data[patient_id][visit_id][feature]
                            if feature == item+'_1':
                                if value == '1':
                                    train_label.append(0)
                            elif feature == item+'_2':
                                if value == '1':
                                    train_label.append(1)
                            elif feature == item+'_3':
                                if value == '1':
                                    train_label.append(2)
                            elif feature == item + '_lost':
                                if value == '1':
                                    pass
                            else:
                                if item == '钠' or item == '钙':
                                    continue
                                print(item)
                                print(feature)
                                raise ValueError('')
                        else:
                            value = white_list_imputed_data[patient_id][visit_id][feature]
                            feature_tuple.append(value)
                    train_feature.append(feature_tuple)
        train_feature = np.array(train_feature, dtype=np.float64)
        test_feature = np.array(test_feature, dtype=np.float64)
        train_label = np.array(train_label, dtype=np.int16)
        if len(test_feature) == 0:
            continue

        clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
        clf = clf.fit(train_feature, train_label)
        print(item)
        test_label = clf.predict(test_feature)
        for i in mapping_dict:
            patient_id, visit_id = mapping_dict[i]
            if test_label[i] == 0:
                target_data[patient_id][visit_id][item+'_1'] = 1
            elif test_label[i] == 1:
                target_data[patient_id][visit_id][item + '_2'] = 1
            elif test_label[i] == 2:
                target_data[patient_id][visit_id][item + '_3'] = 1
            else:
                raise ValueError('')

    pop_feature_list = list()
    for patient_id in white_list_imputed_data:
        for visit_id in white_list_imputed_data[patient_id]:
            for feature in white_list_imputed_data[patient_id][visit_id]:
                if feature.__contains__('lost'):
                    pop_feature_list.append(feature)
            break
        break
    for patient_id in white_list_imputed_data:
        for visit_id in white_list_imputed_data[patient_id]:
            for feature in pop_feature_list:
                white_list_imputed_data[patient_id][visit_id].pop(feature)
    return target_data


def feature_pre_process(origin_data, save_path, white_list):
    missing_dict = missing_rate_calc(origin_data)
    discarded_data = feature_discard(origin_data, missing_dict, white_list)
    imputed_data = feature_imputation(discarded_data)
    feature_list = list()
    for patient_id in origin_data:
        for visit_id in origin_data[patient_id]:
            for key in origin_data[patient_id][visit_id]:
                feature_list.append(key)
            break
        break

    data_to_write = list()
    head = ['patient_id', 'visit_id']
    for item in feature_list:
        head.append(item)
    data_to_write.append(head)

    for patient_id in imputed_data:
        for visit_id in imputed_data[patient_id]:
            row = [patient_id, visit_id]
            for i in range(2, len(head)):
                row.append(imputed_data[patient_id][visit_id][head[i]])
            data_to_write.append(row)

    with open(save_path, 'w', encoding='gbk', newline='') as file:
        csv.writer(file).writerows(data_to_write)

    print('finish')


def main():
    source_data_path = os.path.abspath('..\\..\\resource\\二值化后的长期纵向数据.csv')
    feature_path = os.path.abspath('..\\..\\resource\\预处理中间结果.csv')
    # 丢失率很高但是不能丢失的
    white_list = ['脑利钠肽前体_lost', '肌钙蛋白T_lost', 'C-反应蛋白测定_lost', '射血分数_lost']
    origin_data = read_data(source_data_path)
    feature_pre_process(origin_data, feature_path, white_list)


if __name__ == "__main__":
    main()
