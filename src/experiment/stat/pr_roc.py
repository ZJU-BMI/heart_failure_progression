# encoding=utf-8-sig
import os
import csv
import numpy as np
from itertools import islice
from sklearn import metrics
import matplotlib.pyplot as plt


def main():
    save_path = os.path.abspath('..\\..\\..\\resource\\prediction_result\\best_result')
    vanilla_rnn_model_folder_path = os.path.join(save_path, 'vanilla_rnn_autoencoder_true_gru')
    hawkes_rnn_model_folder_path = os.path.join(save_path, 'fused_hawkes_rnn_autoencoder_true_gru')
    concat_hawkes_rnn_model_folder_path = os.path.join(save_path, 'concat_hawkes_rnn_autoencoder_true_gru')
    base_model_file_path = os.path.join(save_path, 'traditional_ml\\传统模型预测细节.csv')

    time_window_chn = ['一年', '三月']
    event_type_chn = ['癌症', '肾病入院', '肺病', '死亡', '心功能1级', '心功能2级', '心功能3级',
                      '心功能4级', '再血管化手术', '其它']
    time_windows_eng = ['1Y', '3M']
    event_type_eng = ['Cancer', 'Renal Dysfunction', 'Lung disease', 'Death', 'NYHA 1', 'NYHA 2', 'NYHA 3',
                      'NYHA 4', 'Revascularization', 'Others']
    event_chn = list()
    event_eng = list()
    for i in range(len(time_window_chn)):
        for j in range(len(event_type_chn)):
            event_chn.append(time_window_chn[i]+event_type_chn[j])
            event_eng.append(time_windows_eng[i]+'-'+event_type_eng[j])
    map_dict = dict()
    for i in range(len(event_chn)):
        map_dict[event_chn[i]] = event_eng[i]

    base_data = read_base_data(event_chn, base_model_file_path)
    vanilla_result_dict = read_rnn_data(event_chn, vanilla_rnn_model_folder_path)
    hawkes_result_dict = read_rnn_data(event_chn, hawkes_rnn_model_folder_path)
    concat_hawkes_result_dict = read_rnn_data(event_chn, concat_hawkes_rnn_model_folder_path)
    plot_pr(base_data, vanilla_result_dict, hawkes_result_dict, concat_hawkes_result_dict, map_dict, save_path)
    plot_roc(base_data, vanilla_result_dict, hawkes_result_dict, concat_hawkes_result_dict, map_dict, save_path)


def plot_roc(base_data, vanilla_result_dict, hawkes_result_dict, concat_hawkes_result_dict, map_dict, save_path):
    index = 1
    plt.rc('font', family='Times New Roman')
    fig = plt.figure(figsize=(8, 8))
    fig.subplots_adjust(hspace=0, wspace=0)
    for key in base_data:
        base_data_label = base_data[key]['label']
        base_data_prediction = base_data[key]['prediction']
        vanilla_label = vanilla_result_dict[key]['label']
        vanilla_prediction = vanilla_result_dict[key]['prediction']
        hawkes_label = hawkes_result_dict[key]['label']
        hawkes_prediction = hawkes_result_dict[key]['prediction']
        concat_hawkes_label = concat_hawkes_result_dict[key]['label']
        concat_hawkes_prediction = concat_hawkes_result_dict[key]['prediction']

        base_fpr, base_tpr, base_thresholds = metrics.roc_curve(base_data_label, base_data_prediction)
        vanilla_fpr, vanilla_tpr, vanilla_thresholds = metrics.roc_curve(vanilla_label, vanilla_prediction)
        fh_fpr, fh_tpr, fh_thresholds = metrics.roc_curve(hawkes_label, hawkes_prediction)
        ch_fpr, ch_tpr, ch_thresholds = metrics.roc_curve(concat_hawkes_label, concat_hawkes_prediction)

        axs = fig.add_subplot(4, 5, index)
        l1 = axs.plot(base_fpr, base_tpr, color='green', label='LR', linewidth=1)[0]
        l2 = axs.plot(vanilla_fpr, vanilla_tpr, color='blue', label='GRU RNN', linewidth=1)[0]
        l3 = axs.plot(fh_fpr, fh_tpr, color='red', label='FH-RNN', linewidth=1)[0]
        l4 = axs.plot(ch_fpr, ch_tpr, color='orange', label='CH-RNN', linewidth=1)[0]
        axs.set_title('{}'.format(map_dict[key]), fontsize=10, fontweight='bold')

        if index % 5 == 1:
            axs.set_yticks([0.0, 1.0, 1.0])
            axs.set_ylabel('TPR', fontsize=10, fontweight='bold')
        else:
            plt.setp(axs.get_yticklabels(), visible=False)
            plt.setp(axs.get_yaxis(), visible=False)
        if index >= 16:
            axs.set_xticks([0.0, 1.0, 1.0])
            axs.set_xlabel('FPR', fontsize=10, fontweight='bold')
        else:
            plt.setp(axs.get_xticklabels(), visible=False)
            plt.setp(axs.get_xaxis(), visible=False)

        index += 1

    legend = fig.legend([l1, l2, l3, l4],
                        labels=['LR', 'RNN', 'FH-RNN', 'CH-RNN'],
                        borderaxespad=0,
                        ncol=4,
                        fontsize=10,
                        loc='center',
                        bbox_to_anchor=[0.54, 1.03],
                        )
    fig.show()
    fig.savefig(os.path.join(save_path, 'roc_curve'), bbox_inches='tight', bbox_extra_artists=(legend,))


def plot_pr(base_data, vanilla_result_dict, hawkes_result_dict, concat_hawkes_result_dict, map_dict, save_path):
    index = 1
    plt.rc('font', family='Times New Roman')
    fig = plt.figure(figsize=(8, 8))
    fig.subplots_adjust(hspace=0, wspace=0)
    for key in base_data:
        base_data_label = base_data[key]['label']
        base_data_prediction = base_data[key]['prediction']
        vanilla_label = vanilla_result_dict[key]['label']
        vanilla_prediction = vanilla_result_dict[key]['prediction']
        hawkes_label = hawkes_result_dict[key]['label']
        hawkes_prediction = hawkes_result_dict[key]['prediction']
        concat_hawkes_label = concat_hawkes_result_dict[key]['label']
        concat_hawkes_prediction = concat_hawkes_result_dict[key]['prediction']

        """
        base_fpr, base_tpr, base_thresholds = metrics.roc_curve(base_data_label, base_data_prediction)
        vanilla_fpr, vanilla_tpr, vanilla_thresholds = metrics.roc_curve(vanilla_label, vanilla_prediction)
        hawkes_fpr, hawkes_tpr, hawkes_thresholds = metrics.roc_curve(hawkes_label, hawkes_prediction)


        plt.figure()
        lw = 2
        plt.figure(figsize=(10, 10))
        plt.plot(base_fpr, base_tpr, color='green', lw=lw, label='LR')
        plt.plot(vanilla_fpr, vanilla_tpr, color='blue', lw=lw, label='Vanilla RNN')
        plt.plot(hawkes_fpr, hawkes_tpr, color='red', lw=lw, label='Hawkes RNN')
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.title('{} ROC'.format(map_dict[key]), fontsize=30)
        plt.legend(loc="lower right", prop={'size': 30})
        plt.savefig(os.path.join(save_path, '{} ROC'.format(map_dict[key])))
        """

        base_p, base_r, base_thresholds = metrics.precision_recall_curve(base_data_label, base_data_prediction)
        vanilla_p, vanilla_r, _ = metrics.precision_recall_curve(vanilla_label, vanilla_prediction)
        hawkes_p, hawkes_r, _ = metrics.precision_recall_curve(hawkes_label, hawkes_prediction)
        c_hawkes_p, c_hawkes_r, _ = metrics.precision_recall_curve(concat_hawkes_label, concat_hawkes_prediction)

        axs = fig.add_subplot(4, 5, index)
        l1 = axs.plot(base_p, base_r, color='green', label='LR', linewidth=1)[0]
        l2 = axs.plot(vanilla_p, vanilla_r, color='blue', label='GRU RNN', linewidth=1)[0]
        l3 = axs.plot(hawkes_p, hawkes_r, color='red', label='FH-RNN', linewidth=1)[0]
        l4 = axs.plot(c_hawkes_p, c_hawkes_r, color='orange', label='CH-RNN', linewidth=1)[0]
        axs.set_title('{}'.format(map_dict[key]), fontsize=10, fontweight='bold')

        if index % 5 == 1:
            axs.set_yticks([0.0, 1.0, 1.0])
            axs.set_ylabel('Precision', fontsize=10, fontweight='bold')
        else:
            plt.setp(axs.get_yticklabels(), visible=False)
            plt.setp(axs.get_yaxis(), visible=False)
        if index >= 16:
            axs.set_xticks([0.0, 1.0, 1.0])
            axs.set_xlabel('Recall', fontsize=10, fontweight='bold')
        else:
            plt.setp(axs.get_xticklabels(), visible=False)
            plt.setp(axs.get_xaxis(), visible=False)

        index += 1

    legend = fig.legend([l1, l2, l3, l4],
                        labels=['LR', 'RNN', 'FH-RNN', 'CH-RNN'],
                        borderaxespad=0,
                        ncol=4,
                        fontsize=10,
                        loc='center',
                        bbox_to_anchor=[0.54, 1.03],
                        )
    fig.show()
    fig.savefig(os.path.join(save_path, 'pr_curve'), bbox_inches='tight', bbox_extra_artists=(legend,))


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


def read_rnn_data(event_chn, folder_path, skip_line=24):
    result_dict = dict()
    for target in event_chn:
        result_dict[target] = {'label': [], 'prediction': []}
        file_list = os.listdir(folder_path)
        for item in file_list:
            if not (item.__contains__('label') and item.__contains__(target)):
                continue
            file_name = os.path.join(folder_path, item)
            with open(file_name, 'r', encoding='gbk', newline="") as file:
                csv_reader = csv.reader(file)
                for i, line in enumerate(csv_reader):
                    if i < skip_line:
                        continue
                    result_dict[target]['label'].append(float(line[0]))
                    result_dict[target]['prediction'].append(float(line[1]))
    return result_dict


def read_base_data(event_list, file_path):
    # 只读LR
    result_dict = dict()
    for target in event_list:
        result_dict[target] = {'label': [], 'prediction': []}

    with open(file_path, 'r', encoding='gbk', newline="") as file:
        csv_reader = csv.reader(file)
        for line in islice(csv_reader, 1, None):
            if line[3] != 'lr' or not result_dict.__contains__(line[2]):
                continue
            result_dict[line[2]]['label'].append(float(line[5]))
            result_dict[line[2]]['prediction'].append(float(line[6]))
    return result_dict


if __name__ == '__main__':
    main()
