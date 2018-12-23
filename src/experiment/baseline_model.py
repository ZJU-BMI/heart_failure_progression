# encoding=utf-8
# 使用LR, SVM, MLP，RF 三种策略进行事件预测（倒数第二次入院），分别计算AUC值，分别使用五折交叉验证，计算十次
import os
import csv
import random
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


def read_data(feature_path, label_path):
    data_dict = dict()
    data_feature_dict = dict()
    label_dict = dict()
    label_feature_dict = dict()

    with open(feature_path, 'r', encoding='gbk', newline='') as file:
        csv_reader = csv.reader(file)
        head_flag = True
        for line in csv_reader:
            if head_flag:
                for i in range(2, len(line)):
                    data_feature_dict[i] = line[i]
                head_flag = False
                continue

            patient_id = line[0]
            visit_id = line[1]

            if not data_dict.__contains__(patient_id):
                data_dict[patient_id] = dict()
            data_dict[patient_id][visit_id] = dict()

            for i in range(2, len(line)):
                data_dict[patient_id][visit_id][data_feature_dict[i]] = line[i]

    # 读取标签
    with open(label_path, 'r', encoding='gbk', newline='') as file:
        csv_reader = csv.reader(file)
        head_flag = True
        for line in csv_reader:
            if head_flag:
                for i in range(2, len(line)):
                    label_feature_dict[i] = line[i]
                head_flag = False
                continue

            patient_id = line[0]
            visit_id = line[1]

            if not label_dict.__contains__(patient_id):
                label_dict[patient_id] = dict()
            label_dict[patient_id][visit_id] = dict()

            for i in range(2, len(line)):
                label_dict[patient_id][visit_id][label_feature_dict[i]] = line[i]

    # 删除入院时间，数据数值化
    for patient_id in data_dict:
        for visit_id in data_dict[patient_id]:
            data_dict[patient_id][visit_id].pop('时间差')
            for key in data_dict[patient_id][visit_id]:
                data_dict[patient_id][visit_id][key] = float(data_dict[patient_id][visit_id][key])

    # 强制输出有序数列
    output_dict = dict()
    for patient_id in data_dict:
        output_dict[patient_id] = list()
        for i in range(100):
            visit_id = str(i)
            if not (data_dict.__contains__(patient_id) and data_dict[patient_id].__contains__(visit_id)):
                continue
            feature_list = list()
            for key in data_dict[patient_id][visit_id]:
                feature_list.append(data_dict[patient_id][visit_id][key])
            key_dict = label_dict[patient_id][visit_id]
            output_dict[patient_id].append([feature_list, key_dict])

    return output_dict


def reconstruct_data(output_dict):
    # 找到最后一个有效数据，再往前数一个，即为所需数据
    last_visit_dict = dict()
    for patient_id in output_dict:
        last_visit = 0
        for i in range(100):
            if i >= len(output_dict[patient_id]):
                continue
            feature = np.array(output_dict[patient_id][i][0])
            feature_sum = feature.sum()
            if feature_sum > 0:
                last_visit = i
        last_visit_dict[patient_id] = last_visit-1

    # 返回可以直接用于5折交叉验证的数据集
    data_list = list()
    for patient_id in output_dict:
        data_list.append(output_dict[patient_id][last_visit_dict[patient_id]])

    random.shuffle(data_list)
    fold_size = len(data_list) // 5
    data_in_five_split = list()
    data_in_five_split.append(data_list[0: fold_size])
    data_in_five_split.append(data_list[1*fold_size: 2*fold_size])
    data_in_five_split.append(data_list[2*fold_size: 3*fold_size])
    data_in_five_split.append(data_list[3*fold_size: 4*fold_size])
    data_in_five_split.append(data_list[4*fold_size: 5*fold_size])
    return data_in_five_split


def svm(train_feature, train_label, test_feature, test_label):
    clf = LinearSVC(random_state=0, tol=1e-5)
    clf.fit(train_feature, train_label)
    predict_label = clf.predict(test_feature)
    acc = metrics.accuracy_score(test_label, predict_label)
    pre = metrics.precision_score(test_label, predict_label)
    recall = metrics.recall_score(test_label, predict_label)
    f1 = metrics.f1_score(test_label, predict_label)

    pred_score = clf.decision_function(test_feature)
    nominator = 0
    denominator = 0
    for i in range(0, len(test_label)):
        for j in range(0, len(test_label)):
            if test_label[i] == 1 and test_label[j] == 0:
                denominator += 1
                if pred_score[i] > pred_score[j]:
                    nominator += 1
    if denominator == 0:
        auc = 0
    else:
        auc = nominator/denominator

    return auc, acc, pre, recall, f1, pred_score


def lr(train_feature, train_label, test_feature, test_label):
    clf = LogisticRegression(random_state=0, solver='lbfgs')
    clf.fit(train_feature, train_label)
    pred_prob = clf.predict_proba(test_feature)[:, 1]
    pred_label = clf.predict(test_feature)

    positive_sum = 0
    for item in test_label:
        positive_sum += item
    if positive_sum > 0:
        auc = metrics.roc_auc_score(test_label, pred_prob)
    else:
        auc = 0
    acc = metrics.accuracy_score(test_label, pred_label)
    pre = metrics.precision_score(test_label, pred_label)
    recall = metrics.recall_score(test_label, pred_label)
    f1 = metrics.f1_score(test_label, pred_label)

    return auc, acc, pre, recall, f1, pred_prob


def rf(train_feature, train_label, test_feature, test_label):
    clf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
    clf.fit(train_feature, train_label)
    pred_prob = clf.predict_proba(test_feature)[:, 1]
    pred_label = clf.predict(test_feature)

    positive_sum = 0
    for item in test_label:
        positive_sum += item
    if positive_sum > 0:
        auc = metrics.roc_auc_score(test_label, pred_prob)
    else:
        auc = 0
    acc = metrics.accuracy_score(test_label, pred_label)
    pre = metrics.precision_score(test_label, pred_label)
    recall = metrics.recall_score(test_label, pred_label)
    f1 = metrics.f1_score(test_label, pred_label)

    return auc, acc, pre, recall, f1, pred_prob


def mlp(train_feature, train_label, test_feature, test_label):
    clf = MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto', beta_1=0.9, beta_2=0.999,
                        early_stopping=False, epsilon=1e-08, hidden_layer_sizes=(15,), learning_rate='constant',
                        learning_rate_init=0.001, max_iter=200, momentum=0.9, n_iter_no_change=10,
                        nesterovs_momentum=True, power_t=0.5,  random_state=1, shuffle=True, solver='lbfgs', tol=0.0001,
                        validation_fraction=0.1, verbose=False, warm_start=False)
    clf.fit(train_feature, train_label)
    pred_prob = clf.predict_proba(test_feature)[:, 1]
    pred_label = clf.predict(test_feature)

    positive_sum = 0
    for item in test_label:
        positive_sum += item
    if positive_sum > 0:
        auc = metrics.roc_auc_score(test_label, pred_prob)
    else:
        auc = 0
    acc = metrics.accuracy_score(test_label, pred_label)
    pre = metrics.precision_score(test_label, pred_label)
    recall = metrics.recall_score(test_label, pred_label)
    f1 = metrics.f1_score(test_label, pred_label)

    return auc, acc, pre, recall, f1, pred_prob


def main():
    label_path = os.path.abspath('..\\..\\resource\\预处理后的长期纵向数据_标签.csv')
    feature_path = os.path.abspath('..\\..\\resource\\预处理后的长期纵向数据_特征.csv')
    output_dict = read_data(feature_path, label_path)
    time_window = ['一年', '半年', '三月', '两年']
    event_order = ['心功能2级', '心功能1级', '心功能3级', '心功能4级', '再血管化手术',
                   '死亡', '肺病', '肾病入院', '癌症', '糖尿病']
    valid_event_set = set()
    for item_1 in time_window:
        for item_2 in event_order:
            valid_event_set.add(item_1+item_2)

    prediction_label_dict = dict()

    result_list = list()
    result_list.append(['CV_Repeat', 'Test_Fold', 'Method', 'Event', 'AUC', 'ACC', 'PRECISION', 'RECALL', 'F1'])
    for repeat in range(0, 10):
        prediction_label_dict[repeat] = dict()
        data_in_five_split = reconstruct_data(output_dict)
        for i in range(0, 5):
            prediction_label_dict[repeat][i] = dict()
            # 构建五折交叉数据集
            train_feature = list()
            test_feature = list()
            train_label_dict = dict()
            test_label_dict = dict()
            for key in data_in_five_split[0][0][1]:
                train_label_dict[key] = list()
                test_label_dict[key] = list()

            for j in range(0, 5):
                if i != j:
                    for data_tuple in data_in_five_split[j]:
                        train_feature.append(data_tuple[0])
                        for key in data_tuple[1]:
                            train_label_dict[key].append(float(data_tuple[1][key]))
                else:
                    for data_tuple in data_in_five_split[j]:
                        test_feature.append(data_tuple[0])
                        for key in data_tuple[1]:
                            test_label_dict[key].append(float(data_tuple[1][key]))

            # 训练
            for key in train_label_dict:
                if not valid_event_set.__contains__(key):
                    continue
                prediction_label_dict[repeat][i][key] = dict()
                lr_result = lr(train_feature, train_label_dict[key], test_feature, test_label_dict[key])
                auc, acc, pre, recall, f1, pred_prob = lr_result
                result_list.append([repeat, i, 'lr', key, auc, acc, pre, recall, f1])
                label_pred = np.concatenate([np.array(test_label_dict[key])[:, np.newaxis], pred_prob[:, np.newaxis]],
                                            axis=1)
                prediction_label_dict[repeat][i][key]['lr'] = label_pred

                svm_result = svm(train_feature, train_label_dict[key], test_feature, test_label_dict[key],)
                auc, acc, pre, recall, f1, pred_score = svm_result
                result_list.append([repeat, i, 'svm', key, auc, acc, pre, recall, f1])
                label_pred = np.concatenate([np.array(test_label_dict[key])[:, np.newaxis], pred_score[:, np.newaxis]],
                                            axis=1)
                prediction_label_dict[repeat][i][key]['svm'] = label_pred

                mlp_result = mlp(train_feature, train_label_dict[key], test_feature, test_label_dict[key])
                auc, acc, pre, recall, f1, pred_prob = mlp_result
                result_list.append([repeat, i, 'mlp', key, auc, acc, pre, recall, f1])
                label_pred = np.concatenate([np.array(test_label_dict[key])[:, np.newaxis], pred_prob[:, np.newaxis]],
                                            axis=1)
                prediction_label_dict[repeat][i][key]['mlp'] = label_pred

                rf_result = rf(train_feature, train_label_dict[key], test_feature, test_label_dict[key])
                auc, acc, pre, recall, f1, pred_prob = rf_result
                result_list.append([repeat, i, 'rf', key, auc, acc, pre, recall, f1])
                label_pred = np.concatenate([np.array(test_label_dict[key])[:, np.newaxis], pred_prob[:, np.newaxis]],
                                            axis=1)
                prediction_label_dict[repeat][i][key]['rf'] = label_pred

    write_path = os.path.abspath('..\\..\\resource\\prediction_result\\traditional_ml\\传统模型预测综合.csv')
    with open(write_path, 'w', encoding='gbk', newline='') as file:
        csv.writer(file).writerows(result_list)

    data_to_write = list()
    data_to_write.append(['repeat', 'fold', 'task', 'model', 'no', 'label', 'prediction'])
    for repeat in prediction_label_dict:
        for i in prediction_label_dict[repeat]:
            for task in prediction_label_dict[repeat][i]:
                for model in prediction_label_dict[repeat][i][task]:
                    for no in range(len(prediction_label_dict[repeat][i][task][model])):
                        label, prediction = prediction_label_dict[repeat][i][task][model][no]
                        row = [repeat, i, task, model, no, label, prediction]
                        data_to_write.append(row)
    write_path = os.path.abspath('..\\..\\resource\\prediction_result\\traditional_ml\\传统模型预测细节.csv')
    with open(write_path, 'w', encoding='gbk', newline='') as file:
        csv.writer(file).writerows(data_to_write)
    print('finish')


if __name__ == "__main__":
    main()
