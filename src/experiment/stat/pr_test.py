import numpy as np
import os
import csv
file_path = os.path.abspath('..\\,,\\..\\..\\..\\resource\\prediction_result\\traditional_ml\\传统模型预测细节.csv')
label_list = list()
pred_list = list()
with open(file_path, 'r', encoding='gbk', newline='') as file:
    csv_reader = csv.reader(file)
    result_dict = dict()
    for line in csv_reader:
        if not result_dict.__contains__(line[2]):
            result_dict[line[2]] = {'label': list(), 'prediction': list()}
        result_dict[line[2]]['label'].append(line[5])
        result_dict[line[2]]['prediction'].append(line[6])

for key in result_dict:
    result_dict[key]['label'] = np.array( result_dict[key]['label'], dtype=np.int16)
    result_dict[key]['prediction'] = np.array(result_dict[key]['prediction'], dtype=np.float32)

pr_result = dict()
for key in result_dict:
    pr_result[key] = dict()
    for i in range(0, 1, 100):
        pr_result[key][i] = {'precision': 0, 'recall': 0}

for key in pr_result:
    label = result_dict[key]['label'] > 0.5
    prediction = result_dict[key]['prediction']
    threshold_list = [0.01*j for j in range(1, 100)]
    for i in threshold_list:
        pred_pos = prediction > i
        tp = np.sum(label * pred_pos)
        pred_neg = prediction < i
        fn = np.sum(label * pred_neg)
        pred_pos = pred_pos.sum()
        pred_neg = pred_neg.sum()
        if pred_pos == 0:
            precision = 0
        else:
            precision = tp / pred_pos
        if tp + fn == 0:
            recall = 0
        else:
            recall = tp / (tp + fn)
        pr_result[key][i] = {'precision': precision, 'recall': recall}

save_path = os.path.abspath('..\\,,\\..\\..\\..\\resource\\prediction_result\\traditional_ml\\传统模型预测pr.csv')
data_to_write = [['key', 'threshold', 'precision', 'recall']]
for key in pr_result:
    for threshold in pr_result[key]:
        data_to_write.append([key, threshold, pr_result[key][threshold]['precision'],
                              pr_result[key][threshold]['recall']])

with open(save_path, 'w', encoding='gbk', newline='') as file:
    csv.writer(file).writerows(data_to_write)
print('finish')
