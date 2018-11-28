# encoding=utf-8
import csv
import os
import hawkes_process as hawkes
from itertools import islice
import numpy as np
import copy


def get_data(file_path, event_id_dict):
    data_dict = dict()
    with open(file_path, 'r', encoding='gbk', newline='') as file:
        csv_reader = csv.reader(file)
        for line in islice(csv_reader, 1, None):
            patient_id = line[0]
            time_interval = int(line[13])
            event_dict = dict()
            event_dict['糖尿病入院'] = line[2]
            event_dict['肾病入院'] = line[3]
            event_dict['其它'] = line[4]
            event_dict['心功能1级'] = line[5]
            event_dict['心功能2级'] = line[6]
            event_dict['心功能3级'] = line[7]
            event_dict['心功能4级'] = line[8]
            event_dict['死亡'] = line[9]
            event_dict['再血管化手术'] = line[10]
            event_dict['癌症'] = line[11]
            event_dict['肺病'] = line[12]

            if not data_dict.__contains__(patient_id):
                data_dict[patient_id] = list()

            no_flag = 0
            for key in event_id_dict:
                if event_dict[key] == '1':
                    data_dict[patient_id].append([event_id_dict[key], time_interval])
                    no_flag += 1
            if no_flag != 1:
                raise ValueError('error_data')

    for patient_id in data_dict:
        event_list = data_dict[patient_id]
        data_dict[patient_id] = sorted(event_list, key=lambda item: item[1])
    return data_dict


def main():
    file_path = os.path.abspath('..\\..\\resource\\预处理后的长期纵向数据_特征.csv')
    base_intensity_path = os.path.abspath('..\\..\\resource\\hawkes_result\\')
    mutual_intensity_path = os.path.abspath('..\\..\\resource\\hawkes_result\\')
    event_id_dict = dict()
    event_id_dict['糖尿病入院'] = 0
    event_id_dict['肾病入院'] = 1
    event_id_dict['其它'] = 2
    event_id_dict['心功能1级'] = 3
    event_id_dict['心功能2级'] = 4
    event_id_dict['心功能3级'] = 5
    event_id_dict['心功能4级'] = 6
    event_id_dict['死亡'] = 7
    event_id_dict['再血管化手术'] = 8
    event_id_dict['癌症'] = 9
    event_id_dict['肺病'] = 10

    event_full_list = get_data(file_path, event_id_dict)
    hawkes_exp = hawkes.Hawkes(event_full_list, event_full_list, 'exp', 'default', event_count=11, omega=0.006)
    hawkes_exp.optimization(10)

    base_intensity = hawkes_exp.base_intensity
    mutual_intensity = hawkes_exp.mutual_intensity

    np.save(os.path.join(base_intensity_path, 'base.npy'), base_intensity)
    np.save(os.path.join(mutual_intensity_path, 'mutual.npy'), mutual_intensity)

    head = [str(i) for i in range(11)]
    for key in event_id_dict:
        head[event_id_dict[key]] = key

    with open(os.path.join(base_intensity_path, 'mutual.csv'), 'w', encoding='gbk', newline='') as file:
        data_to_write = list()
        data_to_write.append(copy.deepcopy(head))
        row = list()
        for item in base_intensity:
            row.append(item[0])
        data_to_write.append(row)
        csv.writer(file).writerows(data_to_write)
    with open(os.path.join(mutual_intensity_path, 'mutual.csv'), 'w', encoding='gbk', newline='') as file:
        data_to_write = list()
        mutual_head = copy.deepcopy(head)
        mutual_head.insert(0, ' ')
        data_to_write.append(mutual_head)
        for i in range(len(head)):
            row = list()
            row.append(head[i])
            for j in range(len(head)):
                row.append(mutual_intensity[i][j])
            data_to_write.append(row)
        csv.writer(file).writerows(data_to_write)


if __name__ == '__main__':
    main()
