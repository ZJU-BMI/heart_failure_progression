# encoding=utf-8-sig
import os
import csv
import numpy as np
import scipy.stats as st


def main():
    root_path = os.path.abspath('..\\..\\..\\resource\\prediction_result\\')

    # 建立数据模板
    model_list = ['vanilla_rnn', 'time_rnn', 'fuse_time_rnn', 'fuse_hawkes_rnn', 'hawkes_rnn']
    time_window = ['两年', '一年', '半年']
    event_order = ['心功能1级', '心功能2级', '心功能3级', '心功能4级', '再血管化手术',
                   '死亡', '肺病', '糖尿病入院', '肾病入院', '癌症']
    event_list = list()
    for item_1 in time_window:
        for item_2 in event_order:
            event_list.append(item_1+item_2)
    result_dict = dict()
    for model in model_list:
        result_dict[model] = dict()
        for task in event_list:
            result_dict[model][task] = 0

    for model in model_list:
        result_folder = os.path.join(root_path, model)
        file_list = os.listdir(result_folder)
        for file_name in file_list:
            file_path = os.path.join('%s\\%s' % (result_folder, file_name))
            task_type = file_name.split('_')[1]

            auc_list = np.zeros(50)
            with open(file_path, 'r', encoding='gbk', newline='') as file:
                csv_reader = csv.reader(file)
                for i, line in enumerate(csv_reader):
                    if i <= 13 or i >= 64:
                        continue
                    auc_list[i-14] = float(line[3])
                mean = np.mean(auc_list)
                ci = st.t.interval(0.95, len(auc_list) - 1, loc=np.mean(auc_list), scale=st.sem(auc_list))
                ci = mean - ci[0]
                result_dict[model][task_type] = [mean, ci]

    data_to_write = list()
    head = ['']
    for item in model_list:
        head.append(item)
    data_to_write.append(head)
    for event in event_list:
        line = [event]
        for model in model_list:
            mean, ci = result_dict[model][event]
            mean = str(mean)[0: 5]
            ci = str(ci)[0: 5]
            line.append(str(mean)+'±'+str(ci))
        data_to_write.append(line)
    with open(os.path.join(root_path, 'rnn_result.csv'), 'w', encoding='gbk', newline='') as file:
        csv.writer(file).writerows(data_to_write)


if __name__ == '__main__':
    main()
