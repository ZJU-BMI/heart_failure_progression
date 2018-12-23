# encoding=utf-8-sig
import os
import csv
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt


def main():
    root_folder = os.path.abspath('..\\..\\..\\resource\\prediction_result')
    model_folder = os.path.join(root_folder, 'cell_search_lstm')
    file_name = 'prediction_label_一年心功能2级_20181222151442.csv'
    save_name = file_name.split('_')[2]
    file_path = os.path.join(model_folder, file_name)
    result_dict = read_data(file_path)

    label = result_dict['label']
    prediction = result_dict['prediction']
    auc, roc_list = get_roc(label, prediction)
    pr_list = get_pr(label, prediction)

    threshold, fpr, tpr = roc_list
    plt.title('ROC', fontproperties="SimHei")
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(os.path.join(model_folder, 'roc_'+save_name))
    plt.clf()

    precision, recall, thresholds = pr_list
    plt.title('{} PR', fontproperties="SimHei")
    plt.plot(recall, precision, 'b')
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.savefig(os.path.join(model_folder, 'pr_'+save_name))
    plt.clf()


def get_roc(label, prediction):
    label = label.reshape([-1])
    prediction = prediction.reshape([-1])
    auc = metrics.roc_auc_score(label, prediction)
    fpr, tpr, threshold = metrics.roc_curve(label, prediction)
    return auc, [threshold, fpr, tpr]


def get_pr(label, prediction):
    label = label.reshape([-1])
    prediction = prediction.reshape([-1])
    precision, recall, thresholds = metrics.precision_recall_curve(label, prediction)
    return precision, recall, thresholds


def read_data(file_path, skip_line=22):
    result_dict = {'label': list(), 'prediction': list()}
    with open(file_path, 'r', encoding='gbk', newline="") as file:
        csv_reader = csv.reader(file)
        for i, line in enumerate(csv_reader):
            if i < skip_line:
                continue
            result_dict['label'].append(float(line[0]))
            result_dict['prediction'].append(float(line[1]))
    result_dict['label'] = np.array(result_dict['label'])
    result_dict['prediction'] = np.array(result_dict['prediction'])
    return result_dict


if __name__ == '__main__':
    main()
