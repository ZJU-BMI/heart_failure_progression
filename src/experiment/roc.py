# encoding=utf-8-sig
import numpy as np
import os
from sklearn import metrics
import csv
import matplotlib.pyplot as plt


def main():
    label_candidate = ['其它', '再血管化手术', '心功能1级', '心功能2级', '心功能3级', '心功能4级', '死亡', '癌症',
                       '肺病', '肾病入院']
    time_candidate = ['一年', '半年', '两年']
    event_list = list()
    for item_1 in label_candidate:
        for item_2 in time_candidate:
            event_list.append(item_2 + item_1)
    save_folder = os.path.abspath('..\\..\\resource\\prediction_result')
    result_dict = read_data(event_list, save_folder)

    for task in event_list:
        label = result_dict[task]['label']
        prediction = result_dict[task]['prediction']
        auc, roc_list = get_roc(label, prediction)
        write_result(auc, roc_list, save_folder, task)

        threshold, fpr, tpr = roc_list
        plt.title('{} ROC'.format(task), fontproperties="SimHei")
        plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.savefig(os.path.join(save_folder, task))
        plt.clf()


def write_result(auc, roc_list, save_path, task):
    data_to_write = []
    head = ['threshold', 'fpr', 'tpr']
    data_to_write.append(head)
    for item in roc_list:
        data_to_write.append(item)

    file_name = '{}_roc_curve_auc_{:.4f}.csv'.format(task, auc)
    file_name = os.path.join(save_path, file_name)
    with open(file_name, 'w', encoding='gbk', newline='') as file:
        csv.writer(file).writerows(data_to_write)


def get_roc(label, prediction):
    label = label.reshape([-1])
    prediction = prediction.reshape([-1])
    auc = metrics.roc_auc_score(label, prediction)
    fpr, tpr, threshold = metrics.roc_curve(label, prediction)
    return auc, [threshold, fpr, tpr]


def read_data(event_list, save_folder):
    result_dict = dict()
    for task in event_list:
        for j in range(5):
            for i in range(10):
                label_file_name = 'task_{}_test_fold_{}_repeat_{}_label.npy'.format(task, j, i)
                prediction_file_name = 'task_{}_test_fold_{}_repeat_{}_prediction.npy'.format(task, j, i)
                label = np.load(os.path.join(save_folder, label_file_name))
                prediction = np.load(os.path.join(save_folder, prediction_file_name))
                result_dict[task]['label'].append(label)
                result_dict[task]['prediction'].append(prediction)
    for task in event_list:
        result_dict[task]['label'] = np.array(result_dict[task]['label'])
        result_dict[task]['prediction'] = np.array(result_dict[task]['prediction'])
    return result_dict


if __name__ == '__main__':
    main()
