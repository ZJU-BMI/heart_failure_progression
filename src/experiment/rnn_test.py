import tensorflow as tf
from sklearn import metrics
import numpy as np
from read_data import DataSource
from model.rnn_regularization import regularization_rnn_model
from model.rnn_dropout import rnn_drop_model
from model.self_attention import self_attention
from model.vanilla_attention import vanilla_attention
from model.hawkes_attention import hawkes_attention
import os
import csv
import datetime
import data_preprocess.five_validation_data_generation as data_generate


def get_metrics(prediction, label, threshold=0.5):
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


def run_graph(data_source, max_step, node_list, predict_task, test_step_interval, batch_size, hawkes=False,
              task_time_value=None, task_type_value=None):
    test_label = data_source.get_test_label()[predict_task]
    test_feature = data_source.get_test_feature()

    best_result = {'auc': 0, 'acc': 0, 'precision': 0, 'recall': 0, 'f1': 0}
    best_prediction = None
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        current_best_auc = -1
        no_improve_count = 0
        for step in range(max_step):
            batch_x, batch_y = data_source.get_next_batch(predict_task)
            if hawkes:
                loss, prediction, x_placeholder, y_placeholder, batch_size_model, phase_indicator, task_time, \
                    task_type, train_node = node_list
                train_feed_dict = {x_placeholder: batch_x, y_placeholder: batch_y, batch_size_model: batch_size,
                                   phase_indicator: -1, task_time: task_time_value, task_type: task_type_value}
            else:
                loss, prediction, x_placeholder, y_placeholder, batch_size_model, phase_indicator, \
                    train_node = node_list
                train_feed_dict = {x_placeholder: batch_x, y_placeholder: batch_y, batch_size_model: batch_size,
                                   phase_indicator: -1}
            sess.run(train_node, feed_dict=train_feed_dict)
            # Test Performance
            if step % test_step_interval == 0:
                if hawkes:
                    loss, prediction, x_placeholder, y_placeholder, batch_size_model, phase_indicator, task_time, \
                        task_type, train_node = node_list
                    test_feed_dict = {x_placeholder: test_feature, y_placeholder: test_label,
                                      batch_size_model: len(test_feature), phase_indicator: 2,
                                      task_time: task_time_value, task_type: task_type_value}
                else:
                    loss, prediction, x_placeholder, y_placeholder, batch_size_model, phase_indicator, \
                        train_node = node_list
                    test_feed_dict = {x_placeholder: test_feature, y_placeholder: test_label,
                                      batch_size_model: len(test_feature), phase_indicator: 2}

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

                # 性能记录与Early Stop的实现
                if auc > current_best_auc:
                    best_result['auc'] = auc
                    best_result['acc'] = accuracy
                    best_result['precision'] = precision
                    best_result['recall'] = recall
                    best_result['f1'] = f1
                    best_prediction = test_prediction

                    current_best_auc = auc
                    no_improve_count = 0
                else:
                    # 至少要跑四分之一最大循环数才能停止
                    if step < max_step / 4:
                        continue
                    # 超过四分之一后，若运行了max iter的十分之一，没有提升，则终止训练
                    if no_improve_count >= max_step / 10:
                        break
                    else:
                        no_improve_count += 1

    return best_result, best_prediction, test_label


def rnn_regularization_test(experiment_config):
    length = experiment_config['length']
    keep_rate = experiment_config['keep_rate']
    learning_rate = experiment_config['learning_rate']
    num_hidden = experiment_config['num_hidden']
    num_feature = experiment_config['num_feature']

    loss, prediction, x_placeholder, y_placeholder, batch_size_model, phase_indicator = \
        regularization_rnn_model(num_feature=num_feature, num_steps=length, num_hidden=num_hidden, keep_rate=keep_rate)
    train_node = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    node_list = [loss, prediction, x_placeholder, y_placeholder, batch_size_model, phase_indicator, train_node]
    five_fold_validation_default(experiment_config, node_list, model='rnn_regularization')


def rnn_dropout_test(experiment_config):
    length = experiment_config['length']
    keep_rate = experiment_config['keep_rate']
    learning_rate = experiment_config['learning_rate']
    num_hidden = experiment_config['num_hidden']
    num_feature = experiment_config['num_feature']

    loss, prediction, x_placeholder, y_placeholder, batch_size_model, phase_indicator = \
        rnn_drop_model(num_feature=num_feature, num_steps=length, num_hidden=num_hidden, keep_rate=keep_rate)
    train_node = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    node_list = [loss, prediction, x_placeholder, y_placeholder, batch_size_model, phase_indicator, train_node]
    five_fold_validation_default(experiment_config, node_list, model='rnn_dropout')


def rnn_vanilla_attention_test(experiment_config):
    length = experiment_config['length']
    keep_rate = experiment_config['keep_rate']
    learning_rate = experiment_config['learning_rate']
    num_hidden = experiment_config['num_hidden']
    num_feature = experiment_config['num_feature']

    loss, prediction, x_placeholder, y_placeholder, batch_size_model, phase_indicator = \
        vanilla_attention(num_feature=num_feature, num_steps=length, num_hidden=num_hidden, keep_rate=keep_rate)
    train_node = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    node_list = [loss, prediction, x_placeholder, y_placeholder, batch_size_model, phase_indicator, train_node]
    five_fold_validation_default(experiment_config, node_list, model='vanilla_attention')


def rnn_self_attention_test(experiment_config):
    length = experiment_config['length']
    keep_rate = experiment_config['keep_rate']
    learning_rate = experiment_config['learning_rate']
    num_hidden = experiment_config['num_hidden']
    num_feature = experiment_config['num_feature']

    loss, prediction, x_placeholder, y_placeholder, batch_size_model, phase_indicator = \
        self_attention(num_feature=num_feature, num_steps=length, num_hidden=num_hidden, keep_rate=keep_rate)
    train_node = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    node_list = [loss, prediction, x_placeholder, y_placeholder, batch_size_model, phase_indicator, train_node]
    five_fold_validation_default(experiment_config, node_list, model='self_attention')


def hawkes_attention_test(experiment_config):
    length = experiment_config['length']
    keep_rate = experiment_config['keep_rate']
    learning_rate = experiment_config['learning_rate']
    num_hidden = experiment_config['num_hidden']
    num_feature = experiment_config['num_feature']
    mutual_intensity_matrix = np.load(experiment_config['mutual_intensity_path'])
    base_intensity_vector = np.load(experiment_config['base_intensity_path'])

    node_tuple = hawkes_attention(num_feature=num_feature, num_steps=length, num_hidden=num_hidden, keep_rate=keep_rate,
                                  mutual_intensity_matrix=mutual_intensity_matrix, event_count=11,
                                  base_intensity_vector=base_intensity_vector)
    train_node = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(node_tuple[0])
    node_list = list()
    for node in node_tuple:
        node_list.append(node)
    node_list.append(train_node)
    five_fold_validation_default(experiment_config, node_list, model='hawkes_attention')


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
    if model == 'hawkes_attention':
        hawkes_flag = True
    else:
        hawkes_flag = False

    # 五折交叉验证，跑十次
    result_record = dict()
    for i in range(10):
        # 每次均重洗一次数据
        data_generate.main(length)
        for j in range(5):
            # 从洗过的数据中读取数据
            data_source = DataSource(data_folder, data_length=length, test_fold_num=j, batch_size=batch_size,
                                     reserve_time=True)
            for task in event_list:
                if not result_record.__contains__(task):
                    result_record[task] = dict()
                if not result_record[task].__contains__(j):
                    result_record[task][j] = dict()

                if hawkes_flag:
                    task_time = task[0: 2]
                    task_type = task[2:]
                    task_type_value = event_id_dict[task_type]
                    if task_time == '半年':
                        task_time_value = 90
                    elif task_time == '一年':
                        task_time_value = 180
                    elif task_time == '两年':
                        task_time_value = 365
                    else:
                        raise ValueError('')

                    best_result, best_prediction, test_label = \
                        run_graph(data_source=data_source, max_step=max_iter,  node_list=node_list, predict_task=task,
                                  batch_size=batch_size, test_step_interval=test_step_interval, hawkes=hawkes_flag,
                                  task_time_value=task_time_value, task_type_value=task_type_value)
                else:
                    best_result, best_prediction, test_label = \
                        run_graph(data_source=data_source, max_step=max_iter,  node_list=node_list, predict_task=task,
                                  batch_size=batch_size, test_step_interval=test_step_interval)
                print(task)
                print(best_result)
                result_record[task][j][i] = best_result, best_prediction, test_label

    # 保存
    save_folder = os.path.abspath('..\\..\\resource\\prediction_result\\{}'.format(model))
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    for task in result_record:
        for j in result_record[task]:
            for i in result_record[task][j]:
                prediction_file_name = 'task_{}_test_fold_{}_repeat_{}_prediction.npy'.format(task, j, i)
                save_path = os.path.join(save_folder, prediction_file_name)
                best_prediction = result_record[task][j][i][1]
                best_prediction = np.array(best_prediction)
                np.save(save_path, best_prediction)

                label_file_name = 'task_{}_test_fold_{}_repeat_{}_label.npy'.format(task, j, i)
                save_path = os.path.join(save_folder, label_file_name)
                best_label = result_record[task][j][i][2]
                best_label = np.array(best_label)
                np.save(save_path, best_label)

    current_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    save_path = os.path.join(save_folder, 'result_{}.csv'.format(current_time))
    data_to_write = []
    for key in experiment_config:
        data_to_write.append([key, experiment_config[key]])
    head = ['task', 'test_fold', 'repeat', 'auc', 'acc', 'precision', 'recall', 'f1']
    data_to_write.append(head)
    for task in result_record:
        for j in result_record[task]:
            for i in result_record[task][j]:
                row = []
                result = result_record[task][j][i][0]
                row.append(task)
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


def set_hyperparameter(full_event_test=False):
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
    num_hidden = 8
    batch_size = 256
    num_feature = 123
    learning_rate = 0.001
    keep_rate = 0.6
    max_iter = 10000
    test_interval = 20

    if full_event_test:
        label_candidate = ['其它', '再血管化手术', '心功能1级', '心功能2级', '心功能3级', '心功能4级', '死亡', '癌症',
                           '糖尿病入院', '肺病', '肾病入院']
        time_candidate = ['半年', '一年', '两年']
    else:
        label_candidate = ['心功能2级', ]
        time_candidate = ['两年', ]

    event_list = list()
    for item_1 in label_candidate:
        for item_2 in time_candidate:
            event_list.append(item_2 + item_1)

    experiment_configure = {'data_folder': data_folder, 'length': length, 'num_hidden': num_hidden,
                            'batch_size': batch_size, 'num_feature': num_feature, 'learning_rate': learning_rate,
                            'keep_rate': keep_rate, 'max_iter': max_iter, 'test_step_interval': test_interval,
                            'event_list': event_list, 'mutual_intensity_path': mutual_intensity_path,
                            'base_intensity_path': base_intensity_path}
    return experiment_configure


if __name__ == '__main__':
    config = set_hyperparameter(full_event_test=False)
    hawkes_attention_test(config)
    rnn_self_attention_test(config)
    rnn_vanilla_attention_test(config)
    rnn_dropout_test(config)
    rnn_regularization_test(config)
