# encoding=utf-8
import csv
import os


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


def data_impute(data_dict):
    """
    数值型数据填补中位数，分类数据填补众数
    :param data_dict:
    :return:
    """
    feature_attribute_dict = dict()
    # 先扫描数据集，确定每个属性的类型（数值型Or分类型）
    for patient in data_dict:
        for visit_id in data_dict[patient]:
            for feature in data_dict[patient][visit_id]:
                if not feature_attribute_dict.__contains__(feature):
                    feature_attribute_dict[feature] = 'categorical'
                value = data_dict[patient][visit_id][feature]
                if value == '-1' or len(value) > 10:
                    continue
                if float(value) != 1.0 and float(value) != 0.0:
                    feature_attribute_dict[feature] = 'numerical'

    # 得到数据
    feature_dict = dict()
    for feature in feature_attribute_dict:
        feature_dict[feature] = []
    for patient in data_dict:
        for visit_id in data_dict[patient]:
            for feature in data_dict[patient][visit_id]:
                value = data_dict[patient][visit_id][feature]
                if value != '-1':
                    feature_dict[feature].append(float(value))
    for feature in feature_dict:
        feature_dict[feature] = sorted(feature_dict[feature])

    # 得到填充值
    feature_fed_dict = dict()
    for feature in feature_dict:
        median = len(feature_dict[feature]) // 2
        feature_fed_dict[feature] = feature_dict[feature][median]

    # 数据填充
    data_processed_dict = dict()
    for patient_id in data_dict:
        if not data_processed_dict.__contains__(patient_id):
            data_processed_dict[patient_id] = dict()
        for visit_id in data_dict[patient_id]:
            data_processed_dict[patient_id][visit_id] = dict()
            for feature in data_dict[patient_id][visit_id]:
                value = data_dict[patient_id][visit_id][feature]
                if value == '-1':
                    data_processed_dict[patient_id][visit_id][feature] = feature_fed_dict[feature]
                else:
                    data_processed_dict[patient_id][visit_id][feature] = value
    return data_processed_dict


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
                if value == '-1':
                    missing_dict[feature] += 1
    for feature in missing_dict:
        missing_dict[feature] = missing_dict[feature] / row_count
    return missing_dict


def data_normalization(data_dict):
    """
    对除了时间差外的所有数值型数据做Z变换
    :param data_dict:
    :return:
    """
    variance_dict = dict()
    mean_dict = dict()
    feature_attribute_dict = dict()
    # 先扫描数据集，确定每个属性的类型（数值型Or分类型）
    for patient in data_dict:
        for visit_id in data_dict[patient]:
            for feature in data_dict[patient][visit_id]:
                if not feature_attribute_dict.__contains__(feature):
                    feature_attribute_dict[feature] = 'categorical'
                value = data_dict[patient][visit_id][feature]
                if value == '-1':
                    continue
                if float(value) != 1.0 and float(value) != 0.0:
                    feature_attribute_dict[feature] = 'numerical'

    # 计算每个变量的方差
    count = 0
    for patient in data_dict:
        for visit_id in data_dict[patient]:
            count += 1
            for feature in data_dict[patient][visit_id]:
                value = float(data_dict[patient][visit_id][feature])
                if feature_attribute_dict[feature] == 'categorical' or feature == '时间差':
                    continue
                if not variance_dict.__contains__(feature):
                    variance_dict[feature] = value * value
                    mean_dict[feature] = value
                else:
                    variance_dict[feature] += value * value
                    mean_dict[feature] += value
    for feature in mean_dict:
        mean_dict[feature] = mean_dict[feature] / count
        variance_dict[feature] = (variance_dict[feature] / count) ** 0.5

    # z变换
    for patient_id in data_dict:
        for visit_id in data_dict[patient_id]:
            for feature in mean_dict:
                value = float(data_dict[patient_id][visit_id][feature])
                value = (value - mean_dict[feature]) / variance_dict[feature]
                data_dict[patient_id][visit_id][feature] = value
    return data_dict


def feature_pre_process(origin_data, save_path):
    missing_dict = missing_rate_calc(origin_data)
    imputed_data = data_impute(origin_data)
    normalized_data = data_normalization(imputed_data)
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
    missing_list = list()
    for item in head:
        if missing_dict.__contains__(item):
            missing_list.append(missing_dict[item])
        else:
            missing_list.append('None')
    data_to_write.append(missing_list)

    for patient_id in normalized_data:
        for visit_id in normalized_data[patient_id]:
            row = [patient_id, visit_id]
            for i in range(2, len(head)):
                row.append(normalized_data[patient_id][visit_id][head[i]])
            data_to_write.append(row)

    with open(save_path, 'w', encoding='gbk', newline='') as file:
        csv.writer(file).writerows(data_to_write)

    print('finish')


def main():
    source_data_path = os.path.abspath('..\\..\\resource\未预处理长期纵向数据.csv')
    feature_path = os.path.abspath('..\\..\\resource\\预处理中间结果.csv')
    origin_data = read_data(source_data_path)
    feature_pre_process(origin_data, feature_path)


if __name__ == "__main__":
    main()
