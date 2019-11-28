# encoding=utf-8-sig
import csv
import os
from itertools import islice
import numpy as np
import scipy.stats as st


def main():
    file_path = os.path.abspath('..\\..\\..\\resource\\prediction_result\\非时序基线模型结果.csv')
    result_source_dict = dict()
    with open(file_path, 'r', encoding='gbk', newline='') as file:
        csv_reader = csv.reader(file)
        for line in islice(csv_reader, 1, None):
            repeat, fold, method, event, auc, acc, precision, recall, f1 = line
            if not result_source_dict.__contains__(event):
                result_source_dict[event] = dict()
            if not result_source_dict[event].__contains__(method):
                result_source_dict[event][method] = dict()
            if not result_source_dict[event][method].__contains__(repeat):
                result_source_dict[event][method][repeat] = dict()
            result_source_dict[event][method][repeat][fold] = {'auc': auc, 'acc': acc, 'precision': precision,
                                                               'recall': recall, 'f1': f1}

    result_stat_dict = dict()
    for event in result_source_dict:
        if not result_stat_dict.__contains__(event):
            result_stat_dict[event] = dict()
        for method in result_source_dict[event]:
            result_list = list()
            for i in result_source_dict[event][method]:
                for j in result_source_dict[event][method][i]:
                    auc = float(result_source_dict[event][method][i][j]['auc'])
                    if auc != 0.0:
                        result_list.append(auc)
            result_list = np.array(result_list)
            mean = np.mean(result_list)
            ci = st.t.interval(0.95, len(result_list) - 1, loc=np.mean(result_list), scale=st.sem(result_list))
            ci = mean-ci[0]
            result_stat_dict[event][method] = [mean, ci]

    file_path = os.path.abspath('..\\..\\..\\resource\\prediction_result\\非时序基线模型统计.csv')
    head = ['event', 'method', 'auc_mean', 'auc_ci']
    data_to_write = list()
    data_to_write.append(head)
    for event in result_stat_dict:
        for method in result_stat_dict[event]:
            mean, ci = result_stat_dict[event][method]
            data_to_write.append([event, method, mean, ci])
    with open(file_path, 'w', encoding='gbk', newline='') as file:
        csv.writer(file).writerows(data_to_write)


if __name__ == '__main__':
    main()
