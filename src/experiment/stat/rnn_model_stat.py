# encoding=utf-8-sig
import os
import csv
import numpy as np
import scipy.stats as st


def main():
    root_path = os.path.abspath('..\\..\\..\\resource\\prediction_result\\')

    # 建立数据模板
    model_type_list = ['concat_hawkes_rnn_autoencoder_true_gru',
                       'concat_hawkes_rnn_autoencoder_false_gru',
                       'hawkes_rnn_autoencoder_true_gru',
                       'hawkes_rnn_autoencoder_false_gru',
                       'vanilla_rnn_autoencoder_true_gru',
                       'vanilla_rnn_autoencoder_false_gru',
                       # 'cell_search_concat_hawkes_rnn_gru',
                       # 'cell_search_concat_hawkes_rnn_lstm',
                       # 'cell_search_fused_hawkes_rnn_gru',
                       # 'cell_search_fused_hawkes_rnn_lstm',
                       # 'cell_search_vanilla_rnn_gru',
                       # 'cell_search_vanilla_rnn_lstm'
                       ]
    time_window = ['三月', '一年']
    event_order = ['心功能1级', '心功能2级', '心功能3级', '心功能4级', '再血管化手术',
                   '死亡', '肺病', '其它', '肾病入院', '癌症']

    event_list = list()
    for item_1 in time_window:
        for item_2 in event_order:
            event_list.append(item_1+item_2)

    result_dict = dict()
    for model in model_type_list:
        result_dict[model] = dict()

    case_list = dict()
    for model in model_type_list:
        case_list[model] = list()

        result_folder = os.path.join(root_path, model)
        file_list = os.listdir(result_folder)
        for file_name in file_list:
            # 如果文件名不含result，文件记录的是预测细节，不能用
            if not file_name.__contains__('result'):
                continue
            file_path = os.path.join('%s\\%s' % (result_folder, file_name))
            print(file_name)
            task_type = file_name.split('_')[2] + '_' + file_name.split('_')[3] + '_' + file_name.split('_')[4]
            case_list[model].append(task_type)

            auc_list = np.zeros(50)
            with open(file_path, 'r', encoding='gbk', newline='') as file:
                csv_reader = csv.reader(file)
                for i, line in enumerate(csv_reader):
                    if i < 24 or i >= 74:
                        continue
                    auc_list[i-24] = float(line[3])
                mean = np.mean(auc_list)
                ci = st.t.interval(0.95, len(auc_list) - 1, loc=np.mean(auc_list), scale=st.sem(auc_list))
                ci = mean - ci[0]
                result_dict[model][task_type] = [mean, ci]

    data_to_write = list()
    data_to_write.append(['model', 'task', 'performance'])
    for model in model_type_list:
        for event in case_list[model]:
            mean, ci = result_dict[model][event]
            mean = str(mean)[0: 5]
            ci = str(ci)[0: 5]
            line = [model, event, str(mean)+'±'+str(ci)]
            data_to_write.append(line)
    with open(os.path.join(root_path, 'rnn_result.csv'), 'w', encoding='gbk', newline='') as file:
        csv.writer(file).writerows(data_to_write)


if __name__ == '__main__':
    main()
