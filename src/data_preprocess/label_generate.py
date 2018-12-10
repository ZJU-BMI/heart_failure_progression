# encoding=utf-8-sig
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


def generate_label(data_dict, save_path,):
    """
    我们所要关注的是7类指标在半年，1年，2年内是否会发生进展，也就是有14个标签
    :param data_dict:
    :param save_path:
    :return:
    """
    # 构建数据模板
    label_dict = dict()
    for patient_id in data_dict:
        label_dict[patient_id] = list()
        for visit_id in data_dict[patient_id]:
            visit_id_int = int(visit_id)
            time_interval = int(data_dict[patient_id][visit_id]['时间差'])
            event = {'其它': int(data_dict[patient_id][visit_id]['其它']),
                     '肾病入院': int(data_dict[patient_id][visit_id]['肾病入院']),
                     '糖尿病入院': int(data_dict[patient_id][visit_id]['糖尿病入院']),
                     '癌症': int(data_dict[patient_id][visit_id]['癌症']),
                     '肺病': int(data_dict[patient_id][visit_id]['肺病']),
                     '心功能1级': int(data_dict[patient_id][visit_id]['心功能1级']),
                     '心功能2级': int(data_dict[patient_id][visit_id]['心功能2级']),
                     '心功能3级': int(data_dict[patient_id][visit_id]['心功能3级']),
                     '心功能4级': int(data_dict[patient_id][visit_id]['心功能4级']),
                     '再血管化手术': int(data_dict[patient_id][visit_id]['再血管化手术']),
                     '死亡': int(data_dict[patient_id][visit_id]['死亡'])}
            half_year_label = {'其它': 0, '肾病入院': 0, '糖尿病入院': 0, '癌症': 0, '肺病': 0, '心功能1级': 0,
                               '心功能2级': 0, '心功能3级': 0, '心功能4级': 0, '再血管化手术': 0, '死亡': 0}
            one_year_label = {'其它': 0, '肾病入院': 0, '糖尿病入院': 0, '癌症': 0, '肺病': 0, '心功能1级': 0,
                              '心功能2级': 0, '心功能3级': 0, '心功能4级': 0, '再血管化手术': 0, '死亡': 0}
            two_year_label = {'其它': 0, '肾病入院': 0, '糖尿病入院': 0, '癌症': 0, '肺病': 0, '心功能1级': 0,
                              '心功能2级': 0, '心功能3级': 0, '心功能4级': 0, '再血管化手术': 0, '死亡': 0}
            three_month_label = {'其它': 0, '肾病入院': 0, '糖尿病入院': 0, '癌症': 0, '肺病': 0, '心功能1级': 0,
                                 '心功能2级': 0, '心功能3级': 0, '心功能4级': 0, '再血管化手术': 0, '死亡': 0}
            label_dict[patient_id].append([visit_id_int, time_interval, event, three_month_label, half_year_label,
                                           one_year_label, two_year_label])

    # 按照入院id号对数据排序
    for patient_id in label_dict:
        sorted(label_dict[patient_id], key=lambda x: x[0])

    # 对一次入院所对应的标签
    # 统计一年内的所有入院，然后分别把相对应的事件置一，如果没有一年内入院，则标签置零
    # 同样的道理应用于两年内入院
    for patient_id in label_dict:
        for i in range(0, len(label_dict[patient_id])-1):
            # 三个月统计
            for j in range(i+1, len(label_dict[patient_id])):
                event = label_dict[patient_id][j][2]
                time_interval = label_dict[patient_id][j][1] - label_dict[patient_id][i][1]
                if time_interval > 90:
                    continue
                for key in event:
                    if event[key] == 1:
                        label_dict[patient_id][i][3][key] = 1

            # 半年统计
            for j in range(i+1, len(label_dict[patient_id])):
                event = label_dict[patient_id][j][2]
                time_interval = label_dict[patient_id][j][1] - label_dict[patient_id][i][1]
                if time_interval > 180:
                    continue
                for key in event:
                    if event[key] == 1:
                        label_dict[patient_id][i][4][key] = 1

            # 一年统计
            for j in range(i+1, len(label_dict[patient_id])):
                event = label_dict[patient_id][j][2]
                time_interval = label_dict[patient_id][j][1] - label_dict[patient_id][i][1]
                if time_interval > 365:
                    continue
                for key in event:
                    if event[key] == 1:
                        label_dict[patient_id][i][5][key] = 1

            # 两年统计
            for j in range(i+1, len(label_dict[patient_id])):
                event = label_dict[patient_id][j][2]
                time_interval = label_dict[patient_id][j][1] - label_dict[patient_id][i][1]
                if time_interval > 730:
                    continue
                for key in event:
                    if event[key] == 1:
                        label_dict[patient_id][i][6][key] = 1

    data_to_write = list()
    head = ['patient_id', 'visit_id']
    for patient_id in label_dict:
        for key in label_dict[patient_id][0][2]:
            head.append('三月' + key)
        for key in label_dict[patient_id][0][2]:
            head.append('半年' + key)

        for key in label_dict[patient_id][0][2]:
            head.append('一年' + key)
        for key in label_dict[patient_id][0][2]:
            head.append('两年' + key)
        break
    data_to_write.append(head)
    for patient_id in label_dict:
        for item in label_dict[patient_id]:
            visit_id = item[0]
            three_month_event = item[3]
            half_year_event = item[4]
            one_year_event = item[5]
            two_year_event = item[6]
            row = [patient_id, visit_id]
            for key in three_month_event:
                row.append(three_month_event[key])
            for key in half_year_event:
                row.append(half_year_event[key])
            for key in one_year_event:
                row.append(one_year_event[key])
            for key in two_year_event:
                row.append(two_year_event[key])
            data_to_write.append(row)
    with open(save_path, 'w', encoding='gbk', newline='') as file:
        csv.writer(file).writerows(data_to_write)


def main():
    source_data_path = os.path.abspath('..\\..\\resource\\预处理后的长期纵向数据_特征.csv')
    feature_path = os.path.abspath('..\\..\\resource\\预处理后的长期纵向数据_标签.csv')
    origin_data = read_data(source_data_path)
    generate_label(origin_data, feature_path)


if __name__ == '__main__':
    main()
