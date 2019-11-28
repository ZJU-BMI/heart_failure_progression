# encoding=utf-8-sig
import os
import csv
from itertools import islice
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np


def main():
    save_path = os.path.abspath('../../../resource/prediction_result/best_result')
    vanilla_rnn_model_folder_path = os.path.join(save_path, 'vanilla_rnn_autoencoder_true_gru')
    hawkes_rnn_model_folder_path = os.path.join(save_path, 'fused_hawkes_rnn_autoencoder_true_gru')
    concat_hawkes_rnn_model_folder_path = os.path.join(save_path, 'concat_hawkes_rnn_autoencoder_true_gru')
    cnn_model_folder_path = os.path.join(save_path, 'cnn')
    base_model_file_path = os.path.join(save_path, 'traditional_ml/传统模型预测细节.csv')
    hp_365_path = os.path.abspath('../../../resource/hawkes_result/prediction/win_365_fra_None_len_None.npy')
    hp_90_path = os.path.abspath('../../../resource/hawkes_result/prediction/win_90_fra_None_len_None.npy')

    time_window_chn = ['一年', '三月']
    event_type_chn = ['癌症', '肾病入院', '肺病', '死亡', '心功能1级', '心功能2级', '心功能3级',
                      '心功能4级', '再血管化手术']
                      # '其它']
    time_windows_eng = ['1Y', '3M']
    event_type_eng = ['Cancer', 'Renal Dys', 'Lung disease', 'Death', 'NYHA 1', 'NYHA 2', 'NYHA 3',
                      'NYHA 4', 'Revascular', 'Others']
    event_chn = list()
    event_eng = list()
    for i in range(len(time_window_chn)):
        for j in range(len(event_type_chn)):
            event_chn.append(time_window_chn[i]+event_type_chn[j])
            event_eng.append(time_windows_eng[i]+'-'+event_type_eng[j])
    map_dict = dict()
    for i in range(len(event_chn)):
        map_dict[event_chn[i]] = event_eng[i]

    hawkes_data = read_hawkes_process_data(hp_365_path, hp_90_path)
    base_data = read_base_data(event_chn, base_model_file_path)
    vanilla_result_dict = read_rnn_data(event_chn, vanilla_rnn_model_folder_path)
    hawkes_result_dict = read_rnn_data(event_chn, hawkes_rnn_model_folder_path)
    concat_hawkes_result_dict = read_rnn_data(event_chn, concat_hawkes_rnn_model_folder_path)
    cnn_result_dict = read_rnn_data(event_chn, cnn_model_folder_path)
    plot_pr_roc(base_data, hawkes_data, vanilla_result_dict, hawkes_result_dict, concat_hawkes_result_dict,
                cnn_result_dict, map_dict, save_path)


def plot_pr_roc(base_data, hawkes_data, vanilla_result_dict, hawkes_result_dict, concat_hawkes_result_dict,
                cnn_result_dict, map_dict, save_path):
    index = 1
    plt.rc('font', family='Times New Roman')
    event_number = 9
    f, axes = plt.subplots(4, event_number, figsize=(16, 8))

    for key in base_data:
        base_data_label = base_data[key]['label']
        base_data_prediction = base_data[key]['prediction']
        vanilla_label = vanilla_result_dict[key]['label']
        vanilla_prediction = vanilla_result_dict[key]['prediction']
        fused_hawkes_label = hawkes_result_dict[key]['label']
        fused_hawkes_prediction = hawkes_result_dict[key]['prediction']
        concat_hawkes_label = concat_hawkes_result_dict[key]['label']
        concat_hawkes_prediction = concat_hawkes_result_dict[key]['prediction']
        hawkes_prediction = np.reshape(hawkes_data[key]['prediction'], [-1])
        hawkes_label = np.reshape(hawkes_data[key]['label'], [-1])
        cnn_label = cnn_result_dict[key]['label']
        cnn_prediction = cnn_result_dict[key]['prediction']

        base_fpr, base_tpr, base_thresholds = metrics.roc_curve(base_data_label, base_data_prediction)
        vanilla_fpr, vanilla_tpr, vanilla_thresholds = metrics.roc_curve(vanilla_label, vanilla_prediction)
        fh_fpr, fh_tpr, fh_thresholds = metrics.roc_curve(fused_hawkes_label, fused_hawkes_prediction)
        ch_fpr, ch_tpr, ch_thresholds = metrics.roc_curve(concat_hawkes_label, concat_hawkes_prediction)
        hp_fpr, hp_tpr, hp_thresholds = metrics.roc_curve(hawkes_label, hawkes_prediction)
        cnn_fpr, cnn_tpr, cnn_thresholds = metrics.roc_curve(cnn_label, cnn_prediction)

        base_p, base_r, base_thresholds = metrics.precision_recall_curve(base_data_label, base_data_prediction)
        vanilla_p, vanilla_r, _ = metrics.precision_recall_curve(vanilla_label, vanilla_prediction)
        hawkes_p, hawkes_r, _ = metrics.precision_recall_curve(fused_hawkes_label, fused_hawkes_prediction)
        c_hawkes_p, c_hawkes_r, _ = metrics.precision_recall_curve(concat_hawkes_label, concat_hawkes_prediction)
        hp_p, hp_r, _ = metrics.precision_recall_curve(hawkes_label, hawkes_prediction)
        cnn_p, cnn_r, _ = metrics.precision_recall_curve(cnn_label, cnn_prediction)

        # plot figure
        l1 = axes[(index-1)//event_number, (index-1) % event_number].plot(
            base_fpr, base_tpr, color='green', label='LR', linewidth=1)[0]
        l2 = axes[(index-1)//event_number, (index-1) % event_number].plot(
            vanilla_fpr, vanilla_tpr, color='blue', label='RNN', linewidth=1)[0]
        l3 = axes[(index-1)//event_number, (index-1) % event_number].plot(
            fh_fpr, fh_tpr, color='red', label='FH-RNN', linewidth=1)[0]
        l4 = axes[(index-1)//event_number, (index-1) % event_number].plot(
            ch_fpr, ch_tpr, color='orange', label='CH-RNN', linewidth=1)[0]
        l5 = axes[(index - 1) // event_number, (index - 1) % event_number].plot(
            hp_fpr, hp_tpr, color='purple', label='HP', linewidth=1)[0]
        l6 = axes[(index - 1) // event_number, (index - 1) % event_number].plot(
            cnn_fpr, cnn_tpr, color='black', label='CNN', linewidth=1)[0]

        axes[(index - 1) // event_number + 2, (index - 1) % event_number].plot(
            base_r, base_p, color='green', label='LR', linewidth=1)
        axes[(index - 1) // event_number + 2, (index - 1) % event_number].plot(
            vanilla_r, vanilla_p, color='blue', label='RNN', linewidth=1)
        axes[(index - 1) // event_number + 2, (index - 1) % event_number].plot(
            hawkes_r, hawkes_p, color='red', label='FH-RNN', linewidth=1)
        axes[(index - 1) // event_number + 2, (index - 1) % event_number].plot(
            c_hawkes_r, c_hawkes_p, color='orange', label='CH-RNN', linewidth=1)
        axes[(index - 1) // event_number + 2, (index - 1) % event_number].plot(
            hp_r, hp_p, color='purple', label='HP', linewidth=1)
        axes[(index - 1) // event_number + 2, (index - 1) % event_number].plot(
            cnn_r, cnn_p, color='black', label='CNN', linewidth=1)

        # set title
        if index < event_number+1:
            axes[(index - 1) // event_number, (index - 1) % event_number].set_title(
                '{}'.format(map_dict[key].split('-')[1]), fontsize=15, fontweight='bold')
            plt.setp(axes[(index - 1) // event_number + 1, (index - 1) % event_number].get_title(), visible=False)
            plt.setp(axes[(index - 1) // event_number + 2, (index - 1) % event_number].get_title(), visible=False)
            plt.setp(axes[(index - 1) // event_number + 3, (index - 1) % event_number].get_title(), visible=False)

        if index % event_number == 1 and index <= 2*(event_number+1):
            if index <= event_number:
                time = '1Y'
            else:
                time = '3M'
            axes[(index - 1) // event_number, (index - 1) % event_number].set_yticks([0.0, 1.0, 1.0])
            axes[(index - 1) // event_number, (index - 1) % event_number].set_ylabel(
                '{}\nTPR'.format(time), fontsize=15, fontweight='bold')
            axes[(index - 1) // event_number + 2, (index - 1) % event_number].set_yticks([0.0, 1.0, 1.0])
            axes[(index - 1) // event_number + 2, (index - 1) % event_number].\
                set_ylabel('{}\nPrecision'.format(time), fontsize=15, fontweight='bold')
        else:
            plt.setp(axes[(index-1)//event_number + 2, (index-1) % event_number].get_yticklabels(), visible=False)
            plt.setp(axes[(index-1)//event_number + 2, (index-1) % event_number].get_yaxis(), visible=False)
            plt.setp(axes[(index-1)//event_number, (index-1) % event_number].get_yticklabels(), visible=False)
            plt.setp(axes[(index-1)//event_number, (index-1) % event_number].get_yaxis(), visible=False)

        if index <= event_number:
            plt.setp(axes[(index-1)//event_number, (index-1) % event_number].get_xticklabels(), visible=False)
            plt.setp(axes[(index-1)//event_number, (index-1) % event_number].get_xaxis(), visible=False)
            plt.setp(axes[(index - 1) // event_number + 1, (index - 1) % event_number].get_xticklabels(), visible=False)
            plt.setp(axes[(index - 1) // event_number + 1, (index - 1) % event_number].get_xaxis(), visible=False)
            # axes[(index - 1) // event_number + 1, (index - 1) % event_number].set_xticks([0.0, 1.0, 1.0])
            # axes[(index - 1) // event_number + 1, (index - 1) % event_number].
            # set_xlabel('FPR', fontsize=13, fontweight='bold')
        elif index >= event_number+1:
            plt.setp(axes[(index-1)//event_number + 1, (index-1) % event_number].get_xticklabels(), visible=False)
            plt.setp(axes[(index-1)//event_number + 1, (index-1) % event_number].get_xaxis(), visible=False)
            axes[(index - 1) // event_number + 2, (index - 1) % event_number].set_xticks([0.0, 1.0, 1.0])
            axes[(index - 1) // event_number + 2, (index - 1) % event_number].set_xlabel('FPR/Recall', fontsize=15,
                                                                                         fontweight='bold')
        index += 1

    legend = f.legend([l1, l2, l3, l4, l5, l6],
                      labels=['LR', 'RNN', 'FH-RNN', 'CH-RNN', 'HP', 'CNN'],
                      borderaxespad=0,
                      ncol=6,
                      fontsize=15,
                      loc='center',
                      bbox_to_anchor=[0.5, 1.03],)
    f.show()
    f.savefig(os.path.join(save_path, 'roc_curve'), bbox_inches='tight', bbox_extra_artists=(legend,))


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


def read_rnn_data(event_chn, folder_path, skip_line=25):
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


def read_hawkes_process_data(file_365_path, file_90_path):
    # 此处的代码是后期强行加的，凑活着看吧
    result_dict = dict()
    three_month = np.load(file_90_path)
    one_year = np.load(file_365_path)

    result_dict['一年癌症'] = {'label': one_year[:, 10, 1], 'prediction': one_year[:, 10, 0]}
    result_dict['三月癌症'] = {'label': three_month[:, 10, 1], 'prediction': three_month[:, 10, 0]}
    result_dict['一年肾病入院'] = {'label': one_year[:, 1, 1], 'prediction': one_year[:, 1, 0]}
    result_dict['三月肾病入院'] = {'label': three_month[:, 1, 1], 'prediction': three_month[:, 1, 0]}
    result_dict['一年肺病'] = {'label': one_year[:, 9, 1], 'prediction': one_year[:, 9, 0]}
    result_dict['三月肺病'] = {'label': three_month[:, 9, 1], 'prediction': three_month[:, 9, 0]}
    result_dict['一年死亡'] = {'label': one_year[:, 7, 1], 'prediction': one_year[:, 7, 0]}
    result_dict['三月死亡'] = {'label': three_month[:, 7, 1], 'prediction': three_month[:, 7, 0]}
    result_dict['一年心功能1级'] = {'label': one_year[:, 3, 1], 'prediction': one_year[:, 3, 0]}
    result_dict['三月心功能1级'] = {'label': three_month[:, 3, 1], 'prediction': three_month[:, 3, 0]}
    result_dict['一年心功能2级'] = {'label': one_year[:, 4, 1], 'prediction': one_year[:, 4, 0]}
    result_dict['三月心功能2级'] = {'label': three_month[:, 4, 1], 'prediction': three_month[:, 4, 0]}
    result_dict['一年心功能3级'] = {'label': one_year[:, 5, 1], 'prediction': one_year[:, 5, 0]}
    result_dict['三月心功能3级'] = {'label': three_month[:, 5, 1], 'prediction': three_month[:, 5, 0]}
    result_dict['一年心功能4级'] = {'label': one_year[:, 6, 1], 'prediction': one_year[:, 6, 0]}
    result_dict['三月心功能4级'] = {'label': three_month[:, 6, 1], 'prediction': three_month[:, 6, 0]}
    result_dict['一年再血管化手术'] = {'label': one_year[:, 8, 1], 'prediction': one_year[:, 8, 0]}
    result_dict['三月再血管化手术'] = {'label': three_month[:, 8, 1], 'prediction': three_month[:, 8, 0]}
    result_dict['一年其它'] = {'label': one_year[:, 2, 1], 'prediction': one_year[:, 2, 0]}
    result_dict['三月其它'] = {'label': three_month[:, 2, 1], 'prediction': three_month[:, 2, 0]}
    return result_dict


if __name__ == '__main__':
    main()
