import tensorflow as tf
from sklearn import metrics
import numpy as np
from experiment.read_data import DataSource
from model.fuse_hawkes_rnn import fuse_hawkes_rnn_model
from model.fuse_time_rnn import fuse_time_rnn_model
from model.time_rnn import time_rnn_model
from model.hawkes_rnn import hawkes_rnn_model
from model.vanilla_rnn import vanilla_rnn_model
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


def fuse_hawkes_rnn_test(shared_hyperparameter):
    length = shared_hyperparameter['length']
    learning_rate = shared_hyperparameter['learning_rate']
    keep_rate = shared_hyperparameter['keep_rate']
    num_hidden = shared_hyperparameter['num_hidden']
    num_feature = shared_hyperparameter['num_feature']

    g = tf.Graph()
    with g.as_default():
        loss, prediction, x_placeholder, y_placeholder, batch_size, phase_indicator, task_type, time_interval, \
            mutual_intensity, base_intensity = fuse_hawkes_rnn_model(num_feature=num_feature, num_steps=length,
                                                                     num_hidden=num_hidden, keep_rate=keep_rate)
        train_node = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

        node_list = [loss, prediction, x_placeholder, y_placeholder, batch_size, phase_indicator, task_type,
                     time_interval, mutual_intensity, base_intensity, train_node]
        five_fold_validation_default(shared_hyperparameter, node_list, model='fuse_hawkes_rnn')


def fuse_time_rnn_test(shared_hyperparameter):
    length = shared_hyperparameter['length']
    learning_rate = shared_hyperparameter['learning_rate']
    keep_rate = shared_hyperparameter['keep_rate']
    num_hidden = shared_hyperparameter['num_hidden']
    num_feature = shared_hyperparameter['num_feature']

    g = tf.Graph()
    with g.as_default():
        loss, prediction, x_placeholder, y_placeholder, batch_size, phase_indicator, time_interval = \
            fuse_time_rnn_model(num_feature=num_feature, num_steps=length, num_hidden=num_hidden, keep_rate=keep_rate)
        train_node = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

        node_list = [loss, prediction, x_placeholder, y_placeholder, batch_size, phase_indicator, time_interval,
                     train_node]
        five_fold_validation_default(shared_hyperparameter, node_list, model='fuse_time_rnn')


def time_rnn_test(shared_hyperparameter):
    length = shared_hyperparameter['length']
    learning_rate = shared_hyperparameter['learning_rate']
    keep_rate = shared_hyperparameter['keep_rate']
    num_hidden = shared_hyperparameter['num_hidden']
    num_feature = shared_hyperparameter['num_feature']

    g = tf.Graph()
    with g.as_default():
        loss, prediction, x_placeholder, y_placeholder, batch_size, phase_indicator, time_interval = \
            time_rnn_model(num_feature=num_feature, num_steps=length, num_hidden=num_hidden, keep_rate=keep_rate)
        train_node = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

        node_list = [loss, prediction, x_placeholder, y_placeholder, batch_size, phase_indicator, time_interval,
                     train_node]
        five_fold_validation_default(shared_hyperparameter, node_list, model='time_rnn')


def vanilla_rnn_test(shared_hyperparameter):
    length = shared_hyperparameter['length']
    learning_rate = shared_hyperparameter['learning_rate']
    keep_rate = shared_hyperparameter['keep_rate']
    num_hidden = shared_hyperparameter['num_hidden']
    num_feature = shared_hyperparameter['num_feature']

    g = tf.Graph()
    with g.as_default():
        loss, prediction, x_placeholder, y_placeholder, batch_size, phase_indicator = \
            vanilla_rnn_model(num_feature=num_feature, num_steps=length, num_hidden=num_hidden, keep_rate=keep_rate)
        train_node = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

        node_list = [loss, prediction, x_placeholder, y_placeholder, batch_size, phase_indicator, train_node]
        five_fold_validation_default(shared_hyperparameter, node_list, model='vanilla_rnn')


def hawkes_rnn_test(shared_hyperparameter):
    length = shared_hyperparameter['length']
    learning_rate = shared_hyperparameter['learning_rate']
    keep_rate = shared_hyperparameter['keep_rate']
    num_hidden = shared_hyperparameter['num_hidden']
    num_feature = shared_hyperparameter['num_feature']

    g = tf.Graph()
    with g.as_default():
        loss, prediction, x_placeholder, y_placeholder, batch_size, phase_indicator, task_type, time_interval, \
            mutual_intensity, base_intensity = hawkes_rnn_model(num_feature=num_feature, num_steps=length,
                                                                num_hidden=num_hidden, keep_rate=keep_rate)
        train_node = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

        node_list = [loss, prediction, x_placeholder, y_placeholder, batch_size, phase_indicator, task_type,
                     time_interval, mutual_intensity, base_intensity, train_node]
        five_fold_validation_default(shared_hyperparameter, node_list, model='hawkes_rnn')


def run_graph(data_source, max_step, node_list, test_step_interval, task_name, experiment_config, model):

    best_result = {'auc': 0, 'acc': 0, 'precision': 0, 'recall': 0, 'f1': 0}

    mutual_intensity_path = experiment_config['mutual_intensity_path']
    base_intensity_path = experiment_config['base_intensity_path']
    mutual_intensity_value = np.load(mutual_intensity_path)
    base_intensity_value = np.load(base_intensity_path)
    event_id_dict = experiment_config['event_id_dict']
    task_index = event_id_dict[task_name[2:]]

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        current_best_auc = -1
        no_improve_count = 0
        minimum_loss = 10000

        for step in range(max_step):
            train_node = node_list[-1]
            loss = node_list[0]
            prediction = node_list[1]
            train_feed_dict, train_label = feed_dict_and_label(data_object=data_source, node_list=node_list,
                                                               base_intensity_value=base_intensity_value,
                                                               task_index=task_index, test_phase=False, model=model,
                                                               mutual_intensity_value=mutual_intensity_value,
                                                               task_name=task_name)
            sess.run(train_node, feed_dict=train_feed_dict)

            # Test Performance
            if step % test_step_interval == 0:
                test_feed_dict, test_label = feed_dict_and_label(data_object=data_source, node_list=node_list,
                                                                 base_intensity_value=base_intensity_value,
                                                                 task_index=task_index, test_phase=True, model=model,
                                                                 mutual_intensity_value=mutual_intensity_value,
                                                                 task_name=task_name)

                train_loss, train_prediction = sess.run([loss, prediction], feed_dict=train_feed_dict)
                test_loss, test_prediction = sess.run([loss, prediction], feed_dict=test_feed_dict)
                accuracy_, precision_, recall_, f1_, auc_ = get_metrics(train_prediction, train_label)
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
                    # saver.save(sess, save_path)

                if test_loss >= minimum_loss and auc <= current_best_auc:
                    # 连续20次测试Loss或AUC没有优化，则停止训练
                    if no_improve_count >= test_step_interval * 20:
                        break
                    else:
                        no_improve_count += test_step_interval
                else:
                    no_improve_count = 0

                if auc > current_best_auc:
                    current_best_auc = auc
                if test_loss < minimum_loss:
                    minimum_loss = test_loss

    return best_result


def feed_dict_and_label(data_object, node_list, task_name, task_index, test_phase, model, mutual_intensity_value,
                        base_intensity_value):
    event_count = len(mutual_intensity_value)

    if test_phase:
        input_x, input_y = data_object.get_test_feature(), data_object.get_test_label()[task_name]
        input_y = input_y[:, np.newaxis]
        phase_value = 1
        batch_size_value = len(input_x)
        time_interval_value = np.zeros([len(input_x), len(input_x[0])])
        for i in range(len(input_x)):
            for j in range(len(input_x[i])):
                if j == 0:
                    time_interval_value[i][j] = 0
                else:
                    time_interval_value[i][j] = input_x[i][j][event_count] - input_x[i][j-1][event_count]
    else:
        input_x, input_y = data_object.get_next_batch(task_name)
        phase_value = -1
        batch_size_value = len(input_x)
        time_interval_value = np.zeros([len(input_x), len(input_x[0])])
        for i in range(len(input_x)):
            for j in range(len(input_x[i])):
                if j == 0:
                    time_interval_value[i][j] = 0
                else:
                    time_interval_value[i][j] = input_x[i][j][event_count] - input_x[i][j-1][event_count]
    if model == 'fuse_hawkes_rnn':
        _, _, x_placeholder, y_placeholder, batch_size, phase_indicator, task_type, time_interval, mutual_intensity, \
            base_intensity, _ = node_list
        feed_dict = {x_placeholder: input_x, y_placeholder: input_y, batch_size: batch_size_value,
                     task_type: task_index, time_interval: time_interval_value, phase_indicator: phase_value,
                     mutual_intensity: mutual_intensity_value, base_intensity: base_intensity_value}
        return feed_dict, input_y
    elif model == 'fuse_time_rnn':
        _, _, x_placeholder, y_placeholder, batch_size, phase_indicator, time_interval, _ = node_list
        feed_dict = {x_placeholder: input_x, y_placeholder: input_y, batch_size: batch_size_value,
                     time_interval: time_interval_value, phase_indicator: phase_value}
        return feed_dict, input_y
    elif model == 'time_rnn':
        _, _, x_placeholder, y_placeholder, batch_size, phase_indicator, time_interval, _ = node_list
        feed_dict = {x_placeholder: input_x, y_placeholder: input_y, batch_size: batch_size_value,
                     time_interval: time_interval_value, phase_indicator: phase_value}
        return feed_dict, input_y
    elif model == 'vanilla_rnn':
        _, _, x_placeholder, y_placeholder, batch_size, phase_indicator, _ = node_list
        feed_dict = {x_placeholder: input_x, y_placeholder: input_y, batch_size: batch_size_value,
                     phase_indicator: phase_value}
        return feed_dict, input_y
    elif model == 'hawkes_rnn':
        _, _, x_placeholder, y_placeholder, batch_size, phase_indicator, task_type, time_interval, mutual_intensity, \
            base_intensity, _ = node_list
        feed_dict = {x_placeholder: input_x, y_placeholder: input_y, batch_size: batch_size_value,
                     task_type: task_index, time_interval: time_interval_value, phase_indicator: phase_value,
                     mutual_intensity: mutual_intensity_value, base_intensity: base_intensity_value}
        return feed_dict, input_y
    else:
        raise ValueError('invalid model name')


def five_fold_validation_default(experiment_config, node_list, model):
    length = experiment_config['length']
    data_folder = experiment_config['data_folder']
    batch_size = experiment_config['batch_size']
    event_list = experiment_config['event_list']
    max_iter = experiment_config['max_iter']
    test_step_interval = experiment_config['test_step_interval']

    save_folder = os.path.abspath('..\\..\\resource\\prediction_result\\{}'.format(model))

    # 五折交叉验证，跑十次
    for task in event_list:
        result_record = dict()
        for j in range(5):
            if not result_record.__contains__(j):
                result_record[j] = dict()
            for i in range(10):
                # 输出当前实验设置
                print('{}_repeat_{}_fold_{}_task_{}_log'.format(model, i, j, task))
                for key in experiment_config:
                    print(key+': '+str(experiment_config[key]))

                # 从洗过的数据中读取数据
                data_source = DataSource(data_folder, data_length=length, test_fold_num=j, batch_size=batch_size,
                                         reserve_time=True, repeat=i)

                best_result = run_graph(data_source=data_source, max_step=max_iter,  node_list=node_list,
                                        test_step_interval=test_step_interval, model=model,
                                        experiment_config=experiment_config, task_name=task)
                print(task)
                print(best_result)
                result_record[j][i] = best_result

                current_auc_mean = 0
                count = 0
                for q in result_record:
                    for p in result_record[q]:
                        current_auc_mean += result_record[q][p]['auc']
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
            result = result_record[j][i]
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
    num_hidden = 32
    keep_rate = 0.8

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

    if full_event_test:
        label_candidate = ['心功能3级', '心功能2级', '心功能1级', '心功能4级', '再血管化手术', '死亡', '癌症',
                           '糖尿病入院', '肺病', '肾病入院']
    else:
        label_candidate = ['心功能2级', ]

    event_list = list()
    for item in label_candidate:
        event_list.append(time_window + item)

    experiment_configure = {'data_folder': data_folder, 'length': length, 'num_hidden': num_hidden,
                            'batch_size': batch_size, 'num_feature': num_feature, 'learning_rate': learning_rate,
                            'max_iter': max_iter, 'test_step_interval': test_interval, 'keep_rate': keep_rate,
                            'event_list': event_list, 'mutual_intensity_path': mutual_intensity_path,
                            'base_intensity_path': base_intensity_path, 'event_id_dict': event_id_dict}
    return experiment_configure


def main():
    time_window_list = ['三月', '一年']
    test_model = 4
    for item in time_window_list:
        config = set_hyperparameter(full_event_test=True, time_window=item)
        if test_model == 0:
            fuse_hawkes_rnn_test(config)
        elif test_model == 1:
            fuse_time_rnn_test(config)
        elif test_model == 2:
            vanilla_rnn_test(config)
        elif test_model == 3:
            time_rnn_test(config)
        elif test_model == 4:
            hawkes_rnn_test(config)


if __name__ == '__main__':
    main()
