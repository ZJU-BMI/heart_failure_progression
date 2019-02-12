import numpy as np
import os
import csv
from sklearn.metrics import roc_auc_score
from itertools import islice


def read_rnn_result(folder_path, event_list, skip_line=24, length=None, fraction=None):
    """
    :param folder_path:
    :param event_list:
    :param skip_line:
    :param length: 可选筛选条件
    :param fraction: 可选筛选条件
    :return:
    """
    result_dict = dict()
    for target in event_list:
        result_dict[target] = {'label': [], 'prediction': []}
        file_list = os.listdir(folder_path)
        for item in file_list:
            if not (item.__contains__('label') and item.__contains__(target)):
                continue
            if fraction is not None:
                if not item.split('_')[8][0:3] == fraction:
                    continue
            if length is not None:
                if not item.split('_')[1] == length:
                    continue

            file_name = os.path.join(folder_path, item)
            with open(file_name, 'r', encoding='gbk', newline="") as file:
                csv_reader = csv.reader(file)
                for i, line in enumerate(csv_reader):
                    if i < skip_line:
                        continue
                    result_dict[target]['label'].append(float(line[0]))
                    result_dict[target]['prediction'].append(float(line[1]))

    prediction = list()
    label = list()
    for target in result_dict:
        prediction.append(result_dict[target]['prediction'])
        label.append(result_dict[target]['label'])
    prediction = np.transpose(np.array(prediction))
    label = np.transpose(np.array(label))
    return prediction, label


def read_various_length(model, folder_path, event_list):
    length = str(model.split('_')[-1])
    skip_line = 24
    return read_rnn_result(folder_path, event_list, skip_line=skip_line, length=length, fraction=None)


def read_various_fraction(model, folder_path, event_list):
    fraction = str(model.split('_')[-1])
    skip_line = 25
    return read_rnn_result(folder_path, event_list, skip_line=skip_line, length=None, fraction=fraction)


def read_base_data(event_list, file_path):
    # 只读LR
    result_dict = dict()
    for target in event_list:
        result_dict[target] = {'label': [], 'prediction': []}

    with open(file_path, 'r', encoding='gbk', newline="") as file:
        csv_reader = csv.reader(file)
        for line in islice(csv_reader, 1, None):
            if line[3] != 'lr' or (not result_dict.__contains__(line[2])):
                continue
            result_dict[line[2]]['label'].append(float(line[5]))
            result_dict[line[2]]['prediction'].append(float(line[6]))

    prediction = list()
    label = list()
    for target in result_dict:
        prediction.append(np.array(result_dict[target]['prediction']))
        label.append(np.array(result_dict[target]['label']))

    prediction = np.array(prediction)
    label = np.array(label)

    prediction = np.transpose(np.array(prediction))
    label = np.transpose(np.array(label))
    return prediction, label


def get_auc(path_dict, model):
    time_window = ['一年', '三月']
    event_type = ['癌症', '肾病入院', '死亡', '心功能1级', '心功能2级', '心功能3级',
                  '心功能4级', '再血管化手术', '其它']
    auc_dict = dict()
    for item_1 in time_window:
        event_list = list()
        for item_2 in event_type:
            event_list.append(item_1+item_2)
        if model.__contains__('lr'):
            prediction, label = read_base_data(event_list, path_dict[model])
        elif model.__contains__('visit'):
            prediction, label = read_various_length(model, path_dict[model], event_list)
        elif model.__contains__('fraction'):
            prediction, label = read_various_fraction(model, path_dict[model], event_list)
        else:
            prediction, label = read_rnn_result(path_dict[model], event_list)
        micro_auc = roc_auc_score(label, prediction, 'micro')
        macro_auc = roc_auc_score(label, prediction, 'macro')
        auc_dict[item_1] = {'micro_auc': micro_auc, 'macro_auc': macro_auc}
    print('model {} success'.format(model))
    return auc_dict


def path_set(root_folder):
    path_dict = dict()

    for i in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        path_dict['lr_fraction_{}'.format(i)] = \
            os.path.join(root_folder, 'traditional_ml/传统模型预测细节_fraction_{}_length_{}_.csv'.format(i, 'None'))

    path_dict['best_rnn'] = os.path.join(root_folder, 'best_result/vanilla_rnn_autoencoder_true_gru')
    path_dict['best_fused_hawkes'] = os.path.join(root_folder, 'best_result/fused_hawkes_rnn_autoencoder_true_gru')
    path_dict['best_concat_hawkes'] = os.path.join(root_folder, 'best_result/concat_hawkes_rnn_autoencoder_true_gru')
    path_dict['lr'] = os.path.join(root_folder, 'traditional_ml/传统模型预测细节_fraction_1_length_None_.csv')

    path_dict['lstm_concat_hawkes'] = os.path.join(root_folder, 'cell_search_concat_hawkes_rnn_lstm')
    path_dict['lstm_fused_hawkes'] = os.path.join(root_folder, 'cell_search_fused_hawkes_rnn_lstm')
    path_dict['lstm_rnn'] = os.path.join(root_folder, 'cell_search_vanilla_rnn_lstm')

    path_dict['dae_rnn'] = os.path.join(root_folder, 'vanilla_rnn_autoencoder_false_gru')
    path_dict['dae_fused_hawkes'] = os.path.join(root_folder, 'fused_hawkes_rnn_autoencoder_false_gru')
    path_dict['dae_concat_hawkes'] = os.path.join(root_folder, 'concat_hawkes_rnn_autoencoder_false_gru')

    for i in range(3, 10):
        path_dict['lr_length_{}'.format(i)] = \
            os.path.join(root_folder, 'traditional_ml/传统模型预测细节_fraction_{}_length_{}_.csv'.format(1, i))

    for i in range(3, 10):
        path_dict['concat_hawkes_visit_{}'.format(i)] = \
            os.path.join(root_folder, 'concat_hawkes_rnn_autoencoder_true_gru')
        path_dict['fused_hawkes_visit_{}'.format(i)] = \
            os.path.join(root_folder, 'fused_hawkes_rnn_autoencoder_true_gru')
        path_dict['rnn_visit_{}'.format(i)] = \
            os.path.join(root_folder, 'vanilla_rnn_autoencoder_true_gru')

    for i in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        path_dict['concat_hawkes_fraction_{}'.format(i)] = \
            os.path.join(root_folder, 'data_fraction_test_concat_hawkes_rnn_autoencoder_true_gru')
        path_dict['fused_hawkes_fraction_{}'.format(i)] = \
            os.path.join(root_folder, 'data_fraction_test_fused_hawkes_rnn_autoencoder_true_gru')
        path_dict['rnn_fraction_{}'.format(i)] = \
            os.path.join(root_folder, 'data_fraction_test_vanilla_rnn_autoencoder_true_gru')

    return path_dict


def main():
    root_folder = os.path.abspath('../../../resource/prediction_result/')
    path_dict = path_set(root_folder)

    auc_dict = dict()
    for model in path_dict:
        single_auc_dict = get_auc(path_dict, model)
        auc_dict[model] = single_auc_dict

    data_to_write = [['model', 'time_window', 'average_type', 'value']]
    for model in auc_dict:
        for time_window in auc_dict[model]:
            for average_type in auc_dict[model][time_window]:
                data_to_write.append([model, time_window, average_type, auc_dict[model][time_window][average_type]])
    save_path = os.path.join(root_folder, 'average_auc.csv')
    with open(save_path, 'w', encoding='utf-8-sig', newline='') as file:
        csv.writer(file).writerows(data_to_write)


if __name__ == '__main__':
    main()
