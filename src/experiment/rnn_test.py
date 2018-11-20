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


def run_graph(data_source, max_step, node_list, predict_task, test_step_interval, batch_size, save_path, hawkes=False,
              task_time_value=None, task_type_value=None):
    test_label = data_source.get_test_label()[predict_task][:, np.newaxis]
    test_feature = data_source.get_test_feature()

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


def rnn_regularization_test(shared_hyperparameter, keep_rate, num_hidden):
    length = shared_hyperparameter['length']
    learning_rate = shared_hyperparameter['learning_rate']
    num_feature = shared_hyperparameter['num_feature']

    shared_hyperparameter['keep_rate'] = keep_rate
    shared_hyperparameter['num_hidden'] = num_hidden
    shared_hyperparameter['model'] = 'regularization'

    g = tf.Graph()
    with g.as_default():
        loss, prediction, x_placeholder, y_placeholder, batch_size_model, phase_indicator = \
            regularization_rnn_model(num_feature=num_feature, num_steps=length, num_hidden=num_hidden,
                                     keep_rate=keep_rate)
        train_node = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

        node_list = [loss, prediction, x_placeholder, y_placeholder, batch_size_model, phase_indicator, train_node]
        five_fold_validation_default(shared_hyperparameter, node_list, model='rnn_regularization')


def rnn_dropout_test(shared_hyperparameter, keep_rate, num_hidden):
    length = shared_hyperparameter['length']
    learning_rate = shared_hyperparameter['learning_rate']
    num_feature = shared_hyperparameter['num_feature']

    shared_hyperparameter['keep_rate'] = keep_rate
    shared_hyperparameter['num_hidden'] = num_hidden
    shared_hyperparameter['model'] = 'dropout'

    g = tf.Graph()
    with g.as_default():
        loss, prediction, x_placeholder, y_placeholder, batch_size_model, phase_indicator = \
            rnn_drop_model(num_feature=num_feature, num_steps=length, num_hidden=num_hidden, keep_rate=keep_rate)
        train_node = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

        node_list = [loss, prediction, x_placeholder, y_placeholder, batch_size_model, phase_indicator, train_node]
        five_fold_validation_default(shared_hyperparameter, node_list, model='rnn_dropout')


def rnn_vanilla_attention_test(shared_hyperparameter, keep_rate, cell_type, num_hidden):
    length = shared_hyperparameter['length']
    learning_rate = shared_hyperparameter['learning_rate']
    num_feature = shared_hyperparameter['num_feature']
    g = tf.Graph()

    shared_hyperparameter['keep_rate'] = keep_rate
    shared_hyperparameter['num_hidden'] = num_hidden
    shared_hyperparameter['model'] = 'vanilla_attention'

    with g.as_default():
        loss, prediction, x_placeholder, y_placeholder, batch_size_model, phase_indicator = \
            vanilla_attention(num_feature=num_feature, num_steps=length, num_hidden=num_hidden, keep_rate=keep_rate,
                              cell_type=cell_type)
        train_node = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

        node_list = [loss, prediction, x_placeholder, y_placeholder, batch_size_model, phase_indicator, train_node]
        five_fold_validation_default(shared_hyperparameter, node_list, model='vanilla_attention')


def rnn_self_attention_test(shared_hyperparameter, cell_type, keep_rate, num_hidden):
    length = shared_hyperparameter['length']
    learning_rate = shared_hyperparameter['learning_rate']
    num_feature = shared_hyperparameter['num_feature']

    shared_hyperparameter['keep_rate'] = keep_rate
    shared_hyperparameter['num_hidden'] = num_hidden
    shared_hyperparameter['model'] = 'self_attention'

    g = tf.Graph()
    with g.as_default():
        loss, prediction, x_placeholder, y_placeholder, batch_size_model, phase_indicator = \
            self_attention(num_feature=num_feature, num_steps=length, num_hidden=num_hidden, keep_rate=keep_rate,
                           cell_type=cell_type)
        train_node = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

        node_list = [loss, prediction, x_placeholder, y_placeholder, batch_size_model, phase_indicator, train_node]
        five_fold_validation_default(shared_hyperparameter, node_list, model='self_attention')


def hawkes_attention_test(shared_hyperparameter, cell_type, keep_rate, num_hidden):
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
        node_tuple = hawkes_attention(num_feature=num_feature, num_steps=length, num_hidden=num_hidden,
                                      keep_rate=keep_rate, mutual_intensity_matrix=mutual_intensity_matrix,
                                      event_count=11, base_intensity_vector=base_intensity_vector, cell_type=cell_type)
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
    if model == 'hawkes_attention':
        hawkes_flag = True
    else:
        hawkes_flag = False

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

                save_path = os.path.join(save_folder, 'task_{}_repeat_{}_fold_{}_model'.format(task, i, j))
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
                                  task_time_value=task_time_value, task_type_value=task_type_value, save_path=save_path)
                else:
                    best_result, best_prediction, test_label = \
                        run_graph(data_source=data_source, max_step=max_iter,  node_list=node_list, predict_task=task,
                                  batch_size=batch_size, test_step_interval=test_step_interval, save_path=save_path)
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
    num_hidden = 10
    batch_size = 256
    num_feature = 123
    learning_rate = 0.0005
    max_iter = 20000
    test_interval = 20

    if full_event_test:
        label_candidate = ['心功能1级', '心功能2级', '心功能3级', '心功能4级', '再血管化手术', '死亡', '癌症',  '其它',
                           '糖尿病入院', '肺病', '肾病入院']
        label_candidate = ['心功能1级', '心功能2级', '心功能3级', '心功能4级']
    else:
        label_candidate = ['心功能2级', ]

    event_list = list()
    for item in label_candidate:
        event_list.append(time_window + item)

    experiment_configure = {'data_folder': data_folder, 'length': length, 'num_hidden': num_hidden,
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
            hawkes_attention_test(config, cell_type='rnn_dropout', keep_rate=0.6, num_hidden=200)
        elif test_model == 1:
            rnn_self_attention_test(config, cell_type='rnn_dropout', keep_rate=0.6, num_hidden=20)
        elif test_model == 2:
            rnn_dropout_test(config, keep_rate=0.6, num_hidden=20)
        elif test_model == 3:
            rnn_regularization_test(config, keep_rate=1, num_hidden=200)
        elif test_model == 4:
            rnn_vanilla_attention_test(config, cell_type='rnn_dropout', keep_rate=0.6, num_hidden=20)


if __name__ == '__main__':
    main()
