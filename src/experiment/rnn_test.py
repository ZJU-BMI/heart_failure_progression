import tensorflow as tf
from sklearn import metrics
import numpy as np
from experiment.read_data import DataSource
from hawkes_rnn import hawkes_rnn_model
from vanilla_rnn import vanilla_rnn_model
from rnn_cell import RawCell, LSTMCell, GRUCell
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


def vanilla_rnn_test(shared_hyperparameter, cell_type, autoencoder):
    length = shared_hyperparameter['length']
    learning_rate = shared_hyperparameter['learning_rate']
    keep_rate_input = shared_hyperparameter['keep_rate_input']
    keep_rate_hidden = shared_hyperparameter['keep_rate_input']
    num_hidden = shared_hyperparameter['num_hidden']
    context_num = shared_hyperparameter['context_num']
    event_num = shared_hyperparameter['event_num']
    dae_weight = shared_hyperparameter['dae_weight']

    g = tf.Graph()
    with g.as_default():
        phase_indicator = tf.placeholder(tf.int32, shape=[], name="phase_indicator")
        if cell_type == 'gru':
            cell = GRUCell(phase_indicator=phase_indicator, keep_prob=keep_rate_hidden, input_length=autoencoder,
                           num_hidden=num_hidden)
        elif cell_type == 'lstm':
            cell = LSTMCell(phase_indicator=phase_indicator, keep_prob=keep_rate_hidden, input_length=autoencoder,
                            num_hidden=num_hidden)
        elif cell_type == 'raw':
            cell = RawCell(phase_indicator=phase_indicator, keep_prob=keep_rate_hidden, input_length=autoencoder,
                           num_hidden=num_hidden)
        else:
            raise ValueError('cell type invalid')

        loss, prediction, event_placeholder, context_placeholder, y_placeholder, batch_size, phase_indicator = \
            vanilla_rnn_model(num_steps=length, num_hidden=num_hidden, cell=cell, keep_rate_input=keep_rate_input,
                              dae_weight=dae_weight, phase_indicator=phase_indicator, num_context=context_num,
                              num_event=event_num)
        train_node = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

        node_list = [loss, prediction, event_placeholder, context_placeholder, y_placeholder, batch_size,
                     phase_indicator, train_node]
        five_fold_validation_default(shared_hyperparameter, node_list, model='vanilla_rnn')


def hawkes_rnn_test(shared_hyperparameter, cell_type, markov_assumption, autoencoder):
    length = shared_hyperparameter['length']
    learning_rate = shared_hyperparameter['learning_rate']
    keep_rate_input = shared_hyperparameter['keep_rate_input']
    keep_rate_hidden = shared_hyperparameter['keep_rate_input']
    num_hidden = shared_hyperparameter['num_hidden']
    context_num = shared_hyperparameter['context_num']
    event_num = shared_hyperparameter['event_num']
    dae_weight = shared_hyperparameter['dae_weight']

    g = tf.Graph()
    with g.as_default():
        phase_indicator = tf.placeholder(tf.int32, shape=[], name="phase_indicator")
        if cell_type == 'gru':
            cell = GRUCell(phase_indicator=phase_indicator, keep_prob=keep_rate_hidden, input_length=autoencoder,
                           num_hidden=num_hidden)
        elif cell_type == 'lstm':
            cell = LSTMCell(phase_indicator=phase_indicator, keep_prob=keep_rate_hidden, input_length=autoencoder,
                            num_hidden=num_hidden)
        elif cell_type == 'raw':
            cell = RawCell(phase_indicator=phase_indicator, keep_prob=keep_rate_hidden, input_length=autoencoder,
                           num_hidden=num_hidden)
        else:
            raise ValueError('cell type invalid')

        loss, prediction, event_placeholder, context_placeholder, y_placeholder, batch_size, phase_indicator, \
            task_type, time_interval, mutual_intensity, base_intensity = \
            hawkes_rnn_model(num_steps=length, num_hidden=num_hidden, cell=cell, keep_rate_input=keep_rate_input,
                             dae_weight=dae_weight, phase_indicator=phase_indicator, num_context=context_num,
                             num_event=event_num, markov_assumption=markov_assumption, auto_encoder_value=autoencoder)
        train_node = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

        node_list = [loss, prediction, event_placeholder, context_placeholder, y_placeholder, batch_size,
                     phase_indicator, task_type, time_interval, mutual_intensity, base_intensity, train_node]
        five_fold_validation_default(shared_hyperparameter, node_list, model='hawkes_rnn')


def run_graph(data_source, max_step, node_list, validate_step_interval, task_name, experiment_config, model):

    best_result = {'auc': 0, 'acc': 0, 'precision': 0, 'recall': 0, 'f1': 0}

    mutual_intensity_path = experiment_config['mutual_intensity_path']
    base_intensity_path = experiment_config['base_intensity_path']
    # model_path_folder = experiment_config['model_path']
    # model_path = os.path.join(model_path_folder, 'repeat_{}_fold_{}_task_name_{}.ckpt'.format(repeat, fold,
    # task_name))

    mutual_intensity_value = np.load(mutual_intensity_path)
    base_intensity_value = np.load(base_intensity_path)
    event_id_dict = experiment_config['event_id_dict']
    task_index = event_id_dict[task_name[2:]]

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allow_growth = True

    init = tf.initialize_all_variables()

    with tf.Session(config=config) as sess:
        sess.run(init)
        no_improve_count = 0
        minimum_loss = 10000

        for step in range(max_step):
            train_node = node_list[-1]
            loss = node_list[0]
            prediction = node_list[1]
            train_feed_dict, train_label = feed_dict_and_label(data_object=data_source, node_list=node_list,
                                                               base_intensity_value=base_intensity_value,
                                                               task_index=task_index, phase=2, model=model,
                                                               mutual_intensity_value=mutual_intensity_value,
                                                               task_name=task_name)
            sess.run(train_node, feed_dict=train_feed_dict)

            # Test Performance
            if step % validate_step_interval == 0:
                validate_feed_dict, validate_label =\
                    feed_dict_and_label(data_object=data_source, node_list=node_list, task_index=task_index,
                                        base_intensity_value=base_intensity_value, phase=3, model=model,
                                        mutual_intensity_value=mutual_intensity_value, task_name=task_name)

                test_feed_dict, test_label = \
                    feed_dict_and_label(data_object=data_source, node_list=node_list, task_index=task_index,
                                        base_intensity_value=base_intensity_value, phase=1, model=model,
                                        mutual_intensity_value=mutual_intensity_value, task_name=task_name)

                train_loss, train_prediction = sess.run([loss, prediction], feed_dict=train_feed_dict)
                validate_loss, validate_prediction = sess.run([loss, prediction], feed_dict=validate_feed_dict)
                test_loss, test_prediction = sess.run([loss, prediction], feed_dict=test_feed_dict)

                _, _, _, _, auc_train = get_metrics(train_prediction, train_label)
                _, _, _, _, auc_validate = get_metrics(validate_prediction, validate_label)
                accuracy, precision, recall, f1, auc = get_metrics(test_prediction, test_label)
                result_template = 'iter:{}, train: loss:{:.5f}, auc:{:.4f}. validate: loss:{:.5f}, auc:{:.4f}'
                result = result_template.format(step, train_loss, auc_train, validate_loss, auc_validate)
                print(result)

                # 虽然Test Set可以直接被算出，但是不打印，测试过程中无法看到Test Set性能的变化
                # 之所以这么设计，是为了不用频繁的读写模型，提高效率

                # 性能记录与Early Stop的实现，前三次不记录（防止出现训练了还不如不训练的情况）
                if step < validate_step_interval * 3:
                    continue

                if validate_loss >= minimum_loss:
                    # 连续20次测试Loss或AUC没有优化，则停止训练
                    if no_improve_count >= validate_step_interval * 20:
                        break
                    else:
                        no_improve_count += validate_step_interval
                else:
                    no_improve_count = 0
                    minimum_loss = validate_loss

                    best_result['auc'] = auc
                    best_result['acc'] = accuracy
                    best_result['precision'] = precision
                    best_result['recall'] = recall
                    best_result['f1'] = f1

    return best_result


def feed_dict_and_label(data_object, node_list, task_name, task_index, phase, model, mutual_intensity_value,
                        base_intensity_value):
    # phase = 1 测试
    if phase == 1:
        input_event, input_context, input_t = data_object.get_test_feature()
        input_y = data_object.get_test_label()[task_name]
        input_y = input_y[:, np.newaxis]
        phase_value = 1
        batch_size_value = len(input_event)
        time_interval_value = np.zeros([len(input_event), len(input_event[0])])
        for i in range(len(input_event)):
            for j in range(len(input_event[i])):
                if j == 0:
                    time_interval_value[i][j] = 0
                else:
                    time_interval_value[i][j] = input_t[i][j] - input_t[i][j-1]
    # phase == 2 训练
    elif phase == 2:
        input_event, input_context, input_t, input_y = data_object.get_next_batch(task_name)
        phase_value = -1
        batch_size_value = len(input_event)
        time_interval_value = np.zeros([len(input_event), len(input_event[0])])
        for i in range(len(input_event)):
            for j in range(len(input_event[i])):
                if j == 0:
                    time_interval_value[i][j] = 0
                else:
                    time_interval_value[i][j] = input_t[i][j] - input_t[i][j-1]
    # phase == 3 验证
    elif phase == 3:
        input_event, input_context, input_t = data_object.get_validation_feature()
        input_y = data_object.get_validation_label()[task_name]
        input_y = input_y[:, np.newaxis]
        phase_value = 1
        batch_size_value = len(input_event)
        time_interval_value = np.zeros([len(input_event), len(input_event[0])])
        for i in range(len(input_event)):
            for j in range(len(input_event[i])):
                if j == 0:
                    time_interval_value[i][j] = 0
                else:
                    time_interval_value[i][j] = input_t[i][j] - input_t[i][j-1]

    else:
        raise ValueError('invalid phase')

    if model == 'vanilla_rnn':
        _, _, event_placeholder, context_placeholder, y_placeholder, batch_size, phase_indicator, _ = node_list
        feed_dict = {event_placeholder: input_event, y_placeholder: input_y, batch_size: batch_size_value,
                     phase_indicator: phase_value, context_placeholder: input_context}
        return feed_dict, input_y
    elif model == 'hawkes_rnn':
        _, _, event_placeholder, context_placeholder, y_placeholder, batch_size, phase_indicator, task_type, \
            time_interval, mutual_intensity, base_intensity, _ = node_list
        feed_dict = {event_placeholder: input_event, y_placeholder: input_y, batch_size: batch_size_value,
                     time_interval: time_interval_value, phase_indicator: phase_value,
                     context_placeholder: input_context, mutual_intensity: mutual_intensity_value,
                     base_intensity: base_intensity_value, task_type: task_index}
        return feed_dict, input_y
    else:
        raise ValueError('invalid model name')


def five_fold_validation_default(experiment_config, node_list, model):
    length = experiment_config['length']
    data_folder = experiment_config['data_folder']
    batch_size = experiment_config['batch_size']
    event_list = experiment_config['event_list']
    max_iter = experiment_config['max_iter']
    validate_step_interval = experiment_config['validate_step_interval']

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
                data_source = DataSource(data_folder, data_length=length, validate_fold_num=j, batch_size=batch_size,
                                         repeat=i)

                best_result = run_graph(data_source=data_source, max_step=max_iter,  node_list=node_list,
                                        validate_step_interval=validate_step_interval, model=model,
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
    model_save_path = os.path.abspath('..\\..\\resource\\model_cache')

    length = 3
    batch_size = 256
    learning_rate = 0.001
    max_iter = 20000
    validate_step_interval = 20
    num_hidden = 32
    keep_rate_hidden = 1
    keep_rate_input = 0.8
    dae_weight = 1
    context_num = 111
    event_num = 11

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
                            'batch_size': batch_size, 'learning_rate': learning_rate, 'dae_weight': dae_weight,
                            'max_iter': max_iter, 'validate_step_interval': validate_step_interval,
                            'keep_rate_input': keep_rate_input, 'keep_rate_hidden': keep_rate_hidden,
                            'event_list': event_list, 'mutual_intensity_path': mutual_intensity_path,
                            'base_intensity_path': base_intensity_path, 'event_id_dict': event_id_dict,
                            'event_num': event_num, 'context_num': context_num,
                            'model_path': model_save_path}
    return experiment_configure


def main():
    time_window_list = ['三月', '一年']
    test_model = 0
    for item in time_window_list:
        config = set_hyperparameter(full_event_test=True, time_window=item)
        if test_model == 0:
            new_graph = tf.Graph()
            with new_graph.as_default():
                hawkes_rnn_test(config, cell_type='lstm', markov_assumption=False, autoencoder=15)
        elif test_model == 1:
            new_graph = tf.Graph()
            with new_graph.as_default():
                vanilla_rnn_test(config, cell_type='lstm', autoencoder=15)
        else:
            raise ValueError('invalid test model')


if __name__ == '__main__':
    main()
