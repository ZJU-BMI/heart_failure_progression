# encoding=utf-8-sig
import os
import csv
import numpy as np
import scipy.stats as st


def main():
    root_path = os.path.abspath('..\\..\\..\\resource\\prediction_result\\')

    # 建立数据模板
    model_type_list = ['hawkes_rnn', 'vanilla_rnn']
    cell_list = ['lstm']
    auto_encoder = ['true']
    time_window = ['s']
    event_order = ['心功能1级', '心功能2级', '心功能3级', '心功能4级', '再血管化手术',
                   '死亡', '肺病', '糖尿病入院', '肾病入院', '癌症']

    # 名称模板
    folder_list = list()
    for model_name in model_type_list:
        if model_name == 'hawkes_rnn':
            name_template = '{}_autoencoder_{}_{}'
            for encoder in auto_encoder:
                for cell in cell_list:
                    name = name_template.format(model_name, encoder, cell)
                    folder_list.append(name)
        elif model_name == 'vanilla_rnn':
            name_template = '{}_autoencoder_{}_{}'
            for encoder in auto_encoder:
                for cell in cell_list:
                    name = name_template.format(model_name, encoder, cell)
                    folder_list.append(name)
        else:
            raise ValueError('')
    event_list = list()
    for item_1 in time_window:
        for item_2 in event_order:
            event_list.append(item_1+item_2)

    result_dict = dict()
    for model in folder_list:
        result_dict[model] = dict()
        for task in event_list:
            result_dict[model][task] = 0

    for model in folder_list:
        result_folder = os.path.join(root_path, model)
        file_list = os.listdir(result_folder)
        for file_name in file_list:
            # 如果文件名不含result，文件记录的是预测细节，不能用
            if not file_name.__contains__('result'):
                continue
            file_path = os.path.join('%s\\%s' % (result_folder, file_name))
            task_type = file_name.split('_')[1]

            auc_list = np.zeros(50)
            with open(file_path, 'r', encoding='gbk', newline='') as file:
                csv_reader = csv.reader(file)
                for i, line in enumerate(csv_reader):
                    if i < 22 or i >= 72:
                        continue
                    auc_list[i-22] = float(line[3])
                mean = np.mean(auc_list)
                ci = st.t.interval(0.95, len(auc_list) - 1, loc=np.mean(auc_list), scale=st.sem(auc_list))
                ci = mean - ci[0]
                result_dict[model][task_type] = [mean, ci]

    data_to_write = list()
    head = ['']
    for item in folder_list:
        head.append(item)
    data_to_write.append(head)
    for event in event_list:
        line = [event]
        for model in folder_list:
            mean, ci = result_dict[model][event]
            mean = str(mean)[0: 5]
            ci = str(ci)[0: 5]
            line.append(str(mean)+'±'+str(ci))
        data_to_write.append(line)
    with open(os.path.join(root_path, 'rnn_result.csv'), 'w', encoding='gbk', newline='') as file:
        csv.writer(file).writerows(data_to_write)


if __name__ == '__main__':
    main()
