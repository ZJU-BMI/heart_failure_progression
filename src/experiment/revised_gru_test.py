import tensorflow as tf
from sklearn import metrics
import numpy as np
from read_data import DataSource
from model.rnn_regularization_revised import revised_rnn_model
import os
import csv
import datetime


def get_metrics(prediction, label, threshold=0.2):
    pred_label = list()
    for item in prediction:
        if item > threshold:
            pred_label.append(1)
        else:
            pred_label.append(0)
    pred_label = np.array(pred_label)
    if label.sum() == 0:
        return -1, -1, -1, -1, -1
    accuracy = metrics.accuracy_score(label, pred_label)
    precision = metrics.precision_score(label, pred_label)
    recall = metrics.recall_score(label, pred_label)
    f1 = metrics.recall_score(label, pred_label)
    auc = metrics.roc_auc_score(label, prediction)
    return accuracy, precision, recall, f1, auc


def run_graph(data_source, max_step, node_list, predict_task, test_step_interval, batch_size, task_type_value):
    test_label = data_source.get_test_label()[predict_task][:, np.newaxis]
    test_feature = data_source.get_test_feature()
    test_time_interval = np.zeros([test_feature.shape[0], test_feature.shape[1]])
    for i in range(len(test_feature)):
        for j in range(len(test_feature[i])):
            if j == 0:
                test_time_interval[i][j] = 0
            else:
                test_time_interval[i][j] = test_feature[i][j][11] - test_feature[i][j - 1][11]

    best_result = {'auc': 0, 'acc': 0, 'precision': 0, 'recall': 0, 'f1': 0}
    best_prediction = None

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allow_growth = True

    # saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        current_best_auc = -1
        no_improve_count = 0

        minimum_loss = 10000
        for step in range(max_step):
            batch_x, batch_y = data_source.get_next_batch(predict_task)
            # 计算时间差，坐标为i, j指的是第i个样本第j-1次入院到第j次入院的时间差
            time_interval_value = np.zeros([batch_x.shape[0], batch_x.shape[1]])
            for i in range(len(batch_x)):
                for j in range(len(batch_x[i])):
                    if j == 0:
                        time_interval_value[i][j] = 0
                    else:
                        time_interval_value[i][j] = batch_x[i][j][11] - batch_x[i][j-1][11]

            loss, prediction, x_placeholder, y_placeholder, batch_size_model, phase_indicator, task_type, \
                time_interval, train_node = node_list
            train_feed_dict = {x_placeholder: batch_x, y_placeholder: batch_y, batch_size_model: batch_size,
                               phase_indicator: -1, task_type: task_type_value, time_interval: time_interval_value}

            sess.run(train_node, feed_dict=train_feed_dict)
            # Test Performance
            if step % test_step_interval == 0:
                loss, prediction, x_placeholder, y_placeholder, batch_size_model, phase_indicator, task_type, \
                    time_interval, train_node = node_list
                test_feed_dict = {x_placeholder: test_feature, y_placeholder: test_label,
                                  batch_size_model: len(test_feature), phase_indicator: 2, task_type: task_type_value,
                                  time_interval: test_time_interval}

                train_loss, train_prediction = sess.run([loss, prediction], feed_dict=train_feed_dict)
                test_loss, test_prediction = sess.run([loss, prediction], feed_dict=test_feed_dict)
                accuracy_, precision_, recall_, f1_, auc_ = get_metrics(train_prediction, batch_y)
                accuracy, precision, recall, f1, auc = get_metrics(test_prediction, test_label)
                result_template = 'iter:{}, train: loss:{:.5f}, auc:{:.4f}, acc:{:.2f}, precision:{:.2f}, ' \
                                  'recall:{:.2f}, f1:{:.2f}. test: loss:{:.5f}, auc:{:.4f}, acc:{:.2f}, ' \
                                  'precision:{:.2f}, recall:{:.2f}, f1:{:.2f}'
                result = result_template.format(step, train_loss, auc_, accuracy_, precision_, recall_, f1_, test_loss,
                                                auc, accuracy, precision, recall, f1)
                print(result)

                # 性能记录与Early Stop的实现，前三次不记录（防止出现训练了还不如不训练的情况）
                if step < test_step_interval * 3:
                    continue

                if auc > current_best_auc:
                    best_result['auc'] = auc
                    best_result['acc'] = accuracy
                    best_result['precision'] = precision
                    best_result['recall'] = recall
                    best_result['f1'] = f1
                    best_prediction = test_prediction
                    # saver.save(sess, save_path)

                if test_loss > minimum_loss and auc < current_best_auc:
                    # 连续10次测试Loss或AUC没有优化，则停止训练
                    if no_improve_count >= test_step_interval * 10:
                        break
                    else:
                        no_improve_count += test_step_interval
                else:
                    no_improve_count = 0

                if auc > current_best_auc:
                    current_best_auc = auc
                if test_loss < minimum_loss:
                    minimum_loss = test_loss

    return best_result, best_prediction, test_label


def revised_gru_hawkes(shared_hyperparameter, keep_rate, num_hidden):
    length = shared_hyperparameter['length']
    learning_rate = shared_hyperparameter['learning_rate']
    num_feature = shared_hyperparameter['num_feature']
    mutual_intensity_matrix = np.load(shared_hyperparameter['mutual_intensity_path'])
    base_intensity_vector = np.load(shared_hyperparameter['base_intensity_path'])

    shared_hyperparameter['keep_rate'] = keep_rate
    shared_hyperparameter['num_hidden'] = num_hidden
    shared_hyperparameter['model'] = 'hawkes_attention'

    g = tf.Graph()
    with g.as_default():
        node_tuple = revised_rnn_model(num_feature=num_feature, num_steps=length, num_hidden=num_hidden,
                                       keep_rate=keep_rate, base_intensity=base_intensity_vector,
                                       mutual_intensity=mutual_intensity_matrix)
        train_node = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(node_tuple[0])
        node_list = list()
        for node in node_tuple:
            node_list.append(node)
        node_list.append(train_node)
        five_fold_validation_default(shared_hyperparameter, node_list, model='hawkes_attention')


def five_fold_validation_default(experiment_config, node_list, model):
    event_id_dict = dict()
    event_id_dict['糖尿病入院'] = 0
    event_id_dict['肾病入院'] = 1
    event_id_dict['其它'] = 2
    event_id_dict['心功能1级'] = 3
    event_id_dict['心功能2级'] = 4
    event_id_dict['心功能3级'] = 5
    event_id_dict['心功能4级'] = 6
    event_id_dict['死亡'] = 7
    event_id_dict['再血管化手术'] = 8
    event_id_dict['癌症'] = 9
    event_id_dict['肺病'] = 10

    length = experiment_config['length']
    data_folder = experiment_config['data_folder']
    batch_size = experiment_config['batch_size']
    event_list = experiment_config['event_list']
    max_iter = experiment_config['max_iter']
    test_step_interval = experiment_config['test_step_interval']

    # 五折交叉验证，跑十次
    for task in event_list:
        result_record = dict()
        save_folder = os.path.abspath('..\\..\\resource\\prediction_result\\{}'.format(model))
        for i in range(10):
            for j in range(5):
                print('{}_repeat_{}_fold_{}_task_{}_log'.format(model, i, j, task))
                # 从洗过的数据中读取数据
                data_source = DataSource(data_folder, data_length=length, test_fold_num=j, batch_size=batch_size,
                                         reserve_time=True, repeat=i)
                if not result_record.__contains__(j):
                    result_record[j] = dict()

                task_type = task[2:]
                task_type_value = event_id_dict[task_type]
                best_result, best_prediction, test_label = \
                    run_graph(data_source=data_source, max_step=max_iter,  node_list=node_list, predict_task=task,
                              batch_size=batch_size, test_step_interval=test_step_interval,
                              task_type_value=task_type_value)
                print(task)
                print(best_result)
                result_record[j][i] = best_result, best_prediction, test_label

                current_auc_mean = 0
                count = 0
                for q in result_record:
                    for p in result_record[q]:
                        current_auc_mean += result_record[q][p][0]['auc']
                        count += 1
                print('current auc mean = {}'.format(current_auc_mean/count))
        save_result(save_folder, experiment_config, result_record, task)


def save_result(save_folder, experiment_config, result_record, task_name):
    # 保存
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    current_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    save_path = os.path.join(save_folder, 'result_{}_{}.csv'.format(task_name, current_time))
    data_to_write = []
    for key in experiment_config:
        data_to_write.append([key, experiment_config[key]])
    head = ['task', 'test_fold', 'repeat', 'auc', 'acc', 'precision', 'recall', 'f1']
    data_to_write.append(head)
    for j in result_record:
        for i in result_record[j]:
            row = []
            result = result_record[j][i][0]
            row.append(task_name)
            row.append(j)
            row.append(i)
            row.append(result['auc'])
            row.append(result['acc'])
            row.append(result['precision'])
            row.append(result['recall'])
            row.append(result['f1'])
            data_to_write.append(row)
    with open(save_path, 'w', encoding='gbk', newline='') as file:
        csv.writer(file).writerows(data_to_write)


def set_hyperparameter(time_window, full_event_test=False):
    """
    :return:
    """
    data_folder = os.path.abspath('..\\..\\resource\\rnn_data')
    mutual_intensity_path = os.path.abspath('..\\..\\resource\\hawkes_result\\mutual.npy')
    base_intensity_path = os.path.abspath('..\\..\\resource\\hawkes_result\\base.npy')
    """
    Standard Hyperparameter
    length = 3
    num_hidden = 4
    batch_size = 256
    num_feature = 122
    learning_rate = 0.001
    keep_rate = 1
    max_iter = 2000
    test_interval = 20
    """
    length = 3
    batch_size = 256
    num_feature = 123
    learning_rate = 0.0005
    max_iter = 20000
    test_interval = 20

    if full_event_test:
        label_candidate = ['心功能1级', '心功能2级', '心功能3级', '心功能4级', '再血管化手术', '死亡', '癌症',  '其它',
                           '糖尿病入院', '肺病', '肾病入院']
    else:
        label_candidate = ['心功能2级', ]

    event_list = list()
    for item in label_candidate:
        event_list.append(time_window + item)

    experiment_configure = {'data_folder': data_folder, 'length': length,
                            'batch_size': batch_size, 'num_feature': num_feature, 'learning_rate': learning_rate,
                            'max_iter': max_iter, 'test_step_interval': test_interval,
                            'event_list': event_list, 'mutual_intensity_path': mutual_intensity_path,
                            'base_intensity_path': base_intensity_path}
    return experiment_configure


def main():
    time_window_list = ['两年', '一年', '半年']
    test_model = 0
    for item in time_window_list:
        config = set_hyperparameter(full_event_test=True, time_window=item)
        if test_model == 0:
            revised_gru_hawkes(config,  keep_rate=0.8, num_hidden=40)


if __name__ == '__main__':
    main()
