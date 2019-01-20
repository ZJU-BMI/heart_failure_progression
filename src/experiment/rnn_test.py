import tensorflow as tf
from sklearn import metrics
import numpy as np
from experiment.read_data import DataSource
from dynamic_hawkes_rnn import hawkes_rnn_model
from dynamic_vanilla_rnn import vanilla_rnn_model
from dynamic_concat_hawkes_rnn import concat_hawkes_model
from rnn_cell import RawCell, LSTMCell, GRUCell
import os
import csv
import GPyOpt
import datetime
import random


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


def vanilla_rnn_test(shared_hyperparameter, cell_type, autoencoder, model, save_last_hidden_state):
    max_length = shared_hyperparameter['max_sequence_length']
    learning_rate = shared_hyperparameter['learning_rate']
    keep_rate_input = shared_hyperparameter['keep_rate_input']
    keep_rate_hidden = shared_hyperparameter['keep_rate_input']
    num_hidden = shared_hyperparameter['num_hidden']
    context_num = shared_hyperparameter['context_num']
    event_num = shared_hyperparameter['event_num']
    dae_weight = shared_hyperparameter['dae_weight']
    shared_hyperparameter['autoencoder'] = autoencoder
    shared_hyperparameter['model'] = model
    shared_hyperparameter['cell_type'] = cell_type

    if autoencoder > 0:
        input_length = autoencoder
    else:
        input_length = event_num + context_num

    model = model + '_' + cell_type

    g = tf.Graph()
    with g.as_default():
        with tf.name_scope('phase'):
            phase_indicator = tf.placeholder(tf.int32, shape=[], name="phase_indicator")
        if cell_type == 'gru':
            cell = GRUCell(phase_indicator=phase_indicator, keep_prob=keep_rate_hidden, input_length=input_length,
                           num_hidden=num_hidden, name='gru')
        elif cell_type == 'lstm':
            cell = LSTMCell(phase_indicator=phase_indicator, keep_prob=keep_rate_hidden, input_length=input_length,
                            num_hidden=num_hidden, name='lstm')
        elif cell_type == 'raw':
            cell = RawCell(phase_indicator=phase_indicator, keep_prob=keep_rate_hidden, input_length=input_length,
                           num_hidden=num_hidden, name='raw')
        else:
            raise ValueError('cell type invalid')

        loss, prediction, event_placeholder, context_placeholder, y_placeholder, batch_size, phase_indicator, \
            sequence_length, final_state = vanilla_rnn_model(num_steps=max_length, num_hidden=num_hidden, cell=cell,
                                                             keep_rate_input=keep_rate_input, dae_weight=dae_weight,
                                                             phase_indicator=phase_indicator, num_context=context_num,
                                                             embedded_size=autoencoder, num_event=event_num,)
        train_node = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

        node_list = [loss, prediction, event_placeholder, context_placeholder, y_placeholder, batch_size,
                     phase_indicator, sequence_length, final_state, train_node]
        mean_auc = five_fold_validation_default(shared_hyperparameter, node_list, model_type='vanilla_rnn',
                                                model_name=model, save_last_hidden_state=save_last_hidden_state)
    return mean_auc


def hawkes_rnn_test(shared_hyperparameter, cell_type, autoencoder, model, save_last_hidden_state):
    max_length = shared_hyperparameter['max_sequence_length']
    learning_rate = shared_hyperparameter['learning_rate']
    keep_rate_input = shared_hyperparameter['keep_rate_input']
    keep_rate_hidden = shared_hyperparameter['keep_rate_input']
    num_hidden = shared_hyperparameter['num_hidden']
    context_num = shared_hyperparameter['context_num']
    event_num = shared_hyperparameter['event_num']
    dae_weight = shared_hyperparameter['dae_weight']
    shared_hyperparameter['model'] = model
    shared_hyperparameter['cell_type'] = cell_type
    shared_hyperparameter['autoencoder'] = autoencoder

    g = tf.Graph()
    model = model+'_'+cell_type
    if autoencoder > 0:
        input_length = autoencoder
    else:
        input_length = event_num + context_num

    with g.as_default():
        phase_indicator = tf.placeholder(tf.int32, shape=[], name="phase_indicator")
        if cell_type == 'gru':
            cell = GRUCell(phase_indicator=phase_indicator, keep_prob=keep_rate_hidden, input_length=input_length,
                           num_hidden=num_hidden, name='gru')
        elif cell_type == 'lstm':
            cell = LSTMCell(phase_indicator=phase_indicator, keep_prob=keep_rate_hidden, input_length=input_length,
                            num_hidden=num_hidden, name='lstm')
        elif cell_type == 'raw':
            cell = RawCell(phase_indicator=phase_indicator, keep_prob=keep_rate_hidden, input_length=input_length,
                           num_hidden=num_hidden, name='raw')
        else:
            raise ValueError('cell type invalid')

        loss, prediction, event_placeholder, context_placeholder, y_placeholder, batch_size, phase_indicator, \
            base_intensity, mutual_intensity, time_list, task_index, sequence_length, final_state = \
            hawkes_rnn_model(num_steps=max_length, num_hidden=num_hidden, cell=cell, keep_rate_input=keep_rate_input,
                             dae_weight=dae_weight, phase_indicator=phase_indicator, num_context=context_num,
                             num_event=event_num, autoencoder_length=autoencoder)
        train_node = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

        node_list = [loss, prediction, event_placeholder, context_placeholder, y_placeholder, batch_size,
                     phase_indicator,  base_intensity, mutual_intensity, time_list, task_index, sequence_length,
                     final_state, train_node]
        mean_auc = five_fold_validation_default(shared_hyperparameter, node_list, model_type='hawkes_rnn',
                                                model_name=model, save_last_hidden_state=save_last_hidden_state)
    return mean_auc


def concat_hawkes_rnn_test(shared_hyperparameter, cell_type, autoencoder, model, save_last_hidden_state):
    max_length = shared_hyperparameter['max_sequence_length']
    learning_rate = shared_hyperparameter['learning_rate']
    keep_rate_input = shared_hyperparameter['keep_rate_input']
    keep_rate_hidden = shared_hyperparameter['keep_rate_input']
    num_hidden = shared_hyperparameter['num_hidden']
    context_num = shared_hyperparameter['context_num']
    event_num = shared_hyperparameter['event_num']
    dae_weight = shared_hyperparameter['dae_weight']
    shared_hyperparameter['model'] = model
    shared_hyperparameter['cell_type'] = cell_type
    shared_hyperparameter['autoencoder'] = autoencoder

    g = tf.Graph()
    model = model+'_'+cell_type
    if autoencoder > 0:
        context_length = autoencoder
    else:
        context_length = context_num

    with g.as_default():
        phase_indicator = tf.placeholder(tf.int32, shape=[], name="phase_indicator")
        if cell_type == 'gru':
            cell_e = GRUCell(phase_indicator=phase_indicator, keep_prob=keep_rate_hidden, input_length=event_num,
                             num_hidden=num_hidden, name='gru_event')
            cell_c = GRUCell(phase_indicator=phase_indicator, keep_prob=keep_rate_hidden, input_length=context_length,
                             num_hidden=num_hidden, name='gru_context')
        elif cell_type == 'lstm':
            cell_e = LSTMCell(phase_indicator=phase_indicator, keep_prob=keep_rate_hidden, input_length=event_num,
                              num_hidden=num_hidden, name='lstm_event')
            cell_c = LSTMCell(phase_indicator=phase_indicator, keep_prob=keep_rate_hidden, input_length=context_length,
                              num_hidden=num_hidden, name='lstm_context')
        elif cell_type == 'raw':
            cell_e = RawCell(phase_indicator=phase_indicator, keep_prob=keep_rate_hidden, input_length=event_num,
                             num_hidden=num_hidden, name='raw_event')
            cell_c = RawCell(phase_indicator=phase_indicator, keep_prob=keep_rate_hidden, input_length=context_length,
                             num_hidden=num_hidden, name='raw_context')
        else:
            raise ValueError('cell type invalid')

        loss, prediction, event_placeholder, context_placeholder, y_placeholder, batch_size, phase_indicator, \
            base_intensity, mutual_intensity, time_list, task_index, sequence_length, concat_final_state = \
            concat_hawkes_model(num_steps=max_length, num_hidden=num_hidden, keep_rate_input=keep_rate_input,
                                dae_weight=dae_weight, phase_indicator=phase_indicator, num_context=context_num,
                                num_event=event_num, autoencoder_length=autoencoder, cell_context=cell_c,
                                cell_event=cell_e)
        train_node = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

        node_list = [loss, prediction, event_placeholder, context_placeholder, y_placeholder, batch_size,
                     phase_indicator, base_intensity, mutual_intensity, time_list, task_index, sequence_length,
                     concat_final_state, train_node]
        mean_auc = five_fold_validation_default(shared_hyperparameter, node_list, model_type='hawkes_rnn',
                                                model_name=model, save_last_hidden_state=save_last_hidden_state)
        return mean_auc


def run_graph(data_source, max_step, node_list, validate_step_interval, task_name, experiment_config, model):

    best_result = {'auc': 0, 'acc': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'prediction_label': list(),
                   'last_hidden_state': None}

    mutual_intensity_path = experiment_config['mutual_intensity_path']
    base_intensity_path = experiment_config['base_intensity_path']
    mutual_intensity_value = np.load(mutual_intensity_path)
    base_intensity_value = np.load(base_intensity_path)
    event_id_dict = experiment_config['event_id_dict']
    task_index = event_id_dict[task_name[2:]]

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allow_growth = True

    init = tf.global_variables_initializer()
    with tf.Session(config=config) as sess:
        sess.run(init)
        no_improve_count = 0
        minimum_loss = 10000

        for step in range(max_step):
            train_node = node_list[-1]
            last_hidden_state = node_list[-2]
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
                test_loss, test_prediction, test_hidden_state = sess.run([loss, prediction, last_hidden_state],
                                                                         feed_dict=test_feed_dict)

                _, _, _, _, auc_train = get_metrics(train_prediction, train_label)
                _, _, _, _, auc_validate = get_metrics(validate_prediction, validate_label)
                accuracy, precision, recall, f1, auc = get_metrics(test_prediction, test_label)
                result_template = 'iter:{}, train: loss:{:.5f}, auc:{:.4f}. validate: loss:{:.5f}, auc:{:.4f}'
                result = result_template.format(step, train_loss, auc_train, validate_loss, auc_validate)
                print(result)

                # 虽然Test Set可以直接被算出，但是不打印，测试过程中无法看到Test Set性能的变化
                # 因此也就不存在"偷看标签"的问题
                # 之所以这么设计，主要是因为标准的Early Stop模型要不断的存取模型副本，然后训练完了再计算测试集性能
                # 这个跑法有点麻烦，而且我们其实也不是特别需要把模型存起来

                # 性能记录与Early Stop的实现，前三次不记录（防止出现训练了还不如不训练的情况）
                if step < validate_step_interval * 3:
                    continue

                if validate_loss >= minimum_loss:
                    # 连续10次验证集性能无差别，则停止训练
                    if no_improve_count >= validate_step_interval * 10:
                        break
                    else:
                        no_improve_count += validate_step_interval
                else:
                    no_improve_count = 0
                    minimum_loss = validate_loss
                    test_label_prediction = np.concatenate([test_label, test_prediction], axis=1)
                    best_result['auc'] = auc
                    best_result['acc'] = accuracy
                    best_result['precision'] = precision
                    best_result['recall'] = recall
                    best_result['f1'] = f1
                    best_result['prediction_label'] = test_label_prediction
                    best_result['last_hidden_state'] = test_hidden_state

                if validate_label.sum() < 0.5 or test_prediction.sum() < 0.5 or train_label.sum() < 0.5:
                    break

    return best_result


def feed_dict_and_label(data_object, node_list, task_name, task_index, phase, model, mutual_intensity_value,
                        base_intensity_value):
    # phase = 1 测试
    if phase == 1:
        input_event, input_context, input_t, input_sequence_length = data_object.get_test_feature()
        input_y = data_object.get_test_label(task_name)
        input_y = input_y[:, np.newaxis]
        phase_value = 1
        batch_size_value = len(input_event[0])
        time_interval_value = np.zeros([len(input_event[0]), len(input_event)])
        for i in range(len(input_event[0])):
            for j in range(len(input_event)):
                if j == 0:
                    time_interval_value[i][j] = 0
                elif j < input_sequence_length[i]:
                    time_interval_value[i][j] = input_t[i][j] - input_t[i][j-1]
    # phase == 2 训练
    elif phase == 2:
        input_event, input_context, input_t, input_sequence_length, input_y = data_object.get_next_batch(task_name)
        phase_value = -1
        batch_size_value = len(input_event[0])
        time_interval_value = np.zeros([len(input_event[0]), len(input_event)])
        for i in range(len(input_event[0])):
            for j in range(len(input_event)):
                if j == 0:
                    time_interval_value[i][j] = 0
                elif j < input_sequence_length[i]:
                    time_interval_value[i][j] = input_t[i][j] - input_t[i][j-1]
    # phase == 3 验证
    elif phase == 3:
        input_event, input_context, input_t, input_sequence_length = data_object.get_validation_feature()
        input_y = data_object.get_validation_label(task_name)
        input_y = input_y[:, np.newaxis]
        phase_value = 1
        batch_size_value = len(input_event[0])
        time_interval_value = np.zeros([len(input_event[0]), len(input_event)])
        for i in range(len(input_event[0])):
            for j in range(len(input_event)):
                if j == 0:
                    time_interval_value[i][j] = 0
                elif j < input_sequence_length[i]:
                    time_interval_value[i][j] = input_t[i][j] - input_t[i][j-1]

    else:
        raise ValueError('invalid phase')

    if model == 'vanilla_rnn':
        _, _, event_placeholder, context_placeholder, y_placeholder, batch_size, phase_indicator, sequence_length, _,\
            _ = node_list
        feed_dict = {event_placeholder: input_event, y_placeholder: input_y, batch_size: batch_size_value,
                     phase_indicator: phase_value, context_placeholder: input_context,
                     sequence_length: input_sequence_length}
        return feed_dict, input_y
    elif model == 'hawkes_rnn':
        _, _, event_placeholder, context_placeholder, y_placeholder, batch_size,phase_indicator, base_intensity, \
            mutual_intensity, time_list, task_index_, sequence_length, _, _ = node_list
        feed_dict = {event_placeholder: input_event, y_placeholder: input_y, batch_size: batch_size_value,
                     time_list: time_interval_value, phase_indicator: phase_value,
                     context_placeholder: input_context, mutual_intensity: mutual_intensity_value,
                     base_intensity: base_intensity_value, task_index_: task_index,
                     sequence_length: input_sequence_length}
        return feed_dict, input_y
    else:
        raise ValueError('invalid model name')


def five_fold_validation_default(experiment_config, node_list, model_type, model_name, save_last_hidden_state):
    max_length = experiment_config['max_sequence_length']
    data_folder = experiment_config['data_folder']
    batch_size = experiment_config['batch_size']
    event_list = experiment_config['event_list']
    max_iter = experiment_config['max_iter']
    validate_step_interval = experiment_config['validate_step_interval']

    metric_folder = os.path.join(experiment_config['prediction_result_folder'], model_name)

    # 五折交叉验证，跑十次
    best_auc = dict()
    for task in event_list:
        result_record = dict()
        prediction_label_dict = dict()
        last_hidden_state_dict = dict()
        for j in range(2):
            if not result_record.__contains__(j):
                result_record[j] = dict()
                prediction_label_dict[j] = dict()
                last_hidden_state_dict[j] = dict()
            for i in range(3):
                # 输出当前实验设置
                print('{}_repeat_{}_fold_{}_task_{}_log'.format(model_name, i, j, task))
                for key in experiment_config:
                    print(key+': '+str(experiment_config[key]))

                # 从洗过的数据中读取数据
                data_source = DataSource(data_folder, data_length=max_length, validate_fold_num=j,
                                         batch_size=batch_size, repeat=i)

                best_result = run_graph(data_source=data_source, max_step=max_iter,  node_list=node_list,
                                        validate_step_interval=validate_step_interval, model=model_type,
                                        experiment_config=experiment_config, task_name=task)

                prediction_label = best_result.pop('prediction_label')
                last_hidden_state = best_result.pop('last_hidden_state')
                result_record[j][i] = best_result
                prediction_label_dict[j][i] = prediction_label
                last_hidden_state_dict[j][i] = last_hidden_state
                print(task)
                print(best_result)

                # 实时输出当前平均性能
                current_auc_mean = 0
                count = 0
                for q in result_record:
                    for p in result_record[q]:
                        current_auc_mean += result_record[q][p]['auc']
                        count += 1
                print('current auc mean = {}'.format(current_auc_mean/count))

        auc_mean = 0
        for q in result_record:
            for p in result_record[q]:
                auc_mean += result_record[q][p]['auc']
        best_auc[task] = auc_mean/50
        save_result(metric_folder, experiment_config, result_record, prediction_label_dict, last_hidden_state_dict,
                    task, save_last_hidden_state)

    # 最后返回综合AUC作为性能评价指标（其实不太合适，但是就这样吧）
    mean_auc = 0
    for key in best_auc:
        mean_auc += best_auc[key]
    mean_auc = mean_auc/len(best_auc)
    return mean_auc


def save_result(save_folder, experiment_config, result_record, prediction_label_dict, last_hidden_state_dict, task_name,
                save_last_state):

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    current_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    # 存储性能
    max_sequence_length = experiment_config['max_sequence_length']

    # 存储预测与标签
    save_path = os.path.join(save_folder, 'length_{}_prediction_label_{}_{}.csv'.format(max_sequence_length, task_name,
                                                                                        current_time))
    data_to_write = [['label', 'prediction']]
    for key in experiment_config:
        data_to_write.append([key, experiment_config[key]])
    for j in prediction_label_dict:
        for i in prediction_label_dict[j]:
            for k in range(len(prediction_label_dict[j][i])):
                data_to_write.append(prediction_label_dict[j][i][k])
    with open(save_path, 'w', encoding='gbk', newline='') as file:
        csv.writer(file).writerows(data_to_write)

    # 存储最后一个hidden state
    if save_last_state:
        for j in last_hidden_state_dict:
            for i in last_hidden_state_dict[j]:
                last_hidden_state = last_hidden_state_dict[j][i]
                save_path = os.path.join(save_folder, 'last_hidden_state')
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                save_path = os.path.join(save_path, 'last_hidden_length_{}_{}_fold_{}_repeat_{}.npy'.format(
                    max_sequence_length, task_name, j, i, current_time))
                np.save(save_path, last_hidden_state)

    # 存储综合性能
    save_path = os.path.join(save_folder, 'result_length_{}_{}_{}.csv'.format(max_sequence_length, task_name,
                                                                              current_time))
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


def set_hyperparameter(time_window, max_sequence_length=10, batch_size=256, dae_weight=1.0,
                       learning_rate=0.001, num_hidden=32, keep_rate_hidden=1, keep_rate_input=0.8,
                       event_list=None):
    """
    :return:
    """
    data_folder = os.path.abspath('../../resource/rnn_data')
    mutual_intensity_path = os.path.abspath('../../resource/hawkes_result/mutual.npy')
    base_intensity_path = os.path.abspath('../../resource/hawkes_result/base.npy')
    model_save_path = os.path.abspath('../../resource/model_cache')
    model_graph_save_path = os.path.abspath('../../resource/model_diagram')
    prediction_metric_folder = os.path.abspath('../../resource/prediction_metric/')
    prediction_result_folder = os.path.abspath('../../resource/prediction_result/')

    max_iter = 10000
    validate_step_interval = 20
    context_num = 189
    event_num = 11

    max_sequence_length = max_sequence_length
    batch_size = batch_size
    learning_rate = learning_rate
    num_hidden = num_hidden
    # cell内drop
    keep_rate_hidden = keep_rate_hidden
    # dae的keep prob
    keep_rate_input = keep_rate_input
    dae_weight = dae_weight

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

    if event_list is None:
        label_candidate = ['心功能2级', '心功能3级', '心功能1级', '心功能4级', '再血管化手术', '死亡', '癌症',
                           '糖尿病入院', '肺病', '肾病入院']
    else:
        label_candidate = event_list

    event_list = list()
    for item in label_candidate:
        event_list.append(time_window + item)

    experiment_configure = {'data_folder': data_folder, 'max_sequence_length': max_sequence_length,
                            'batch_size': batch_size, 'learning_rate': learning_rate, 'dae_weight': dae_weight,
                            'max_iter': max_iter, 'validate_step_interval': validate_step_interval,
                            'keep_rate_input': keep_rate_input, 'keep_rate_hidden': keep_rate_hidden,
                            'event_list': event_list, 'mutual_intensity_path': mutual_intensity_path,
                            'base_intensity_path': base_intensity_path, 'event_id_dict': event_id_dict,
                            'event_num': event_num, 'context_num': context_num, 'num_hidden': num_hidden,
                            'prediction_metric_folder': prediction_metric_folder,
                            'model_path': model_save_path, 'prediction_result_folder': prediction_result_folder,
                            'model_graph_save_path': model_graph_save_path}
    return experiment_configure


def length_test(test_model, event_list):
    for i in range(9, 2, -1):
        performance_test(data_length=i, test_model=test_model, event_list=event_list, save_last_hidden_state=False)


def performance_test(data_length, test_model, save_last_hidden_state, event_list=None):
    time_window_list = ['一年', '三月']
    for cell_type in ['gru']:
        for item in time_window_list:
            if item == '一年':
                max_sequence_length = data_length
                batch_size = 512
                learning_rate = 0.001
                num_hidden = 64
                autoencoder = 16
                keep_rate_hidden = 1
                keep_rate_input = 0.8
                dae_weight = 0.15
            elif item == '三月':
                max_sequence_length = data_length
                batch_size = 128
                learning_rate = 0.008
                num_hidden = 64
                autoencoder = 32
                keep_rate_hidden = 1
                keep_rate_input = 0.8
                dae_weight = 0.15
            else:
                raise ValueError('')
            config = set_hyperparameter(time_window=item, batch_size=batch_size, event_list=event_list,
                                        max_sequence_length=max_sequence_length, learning_rate=learning_rate,
                                        num_hidden=num_hidden, keep_rate_input=keep_rate_input,
                                        keep_rate_hidden=keep_rate_hidden, dae_weight=dae_weight)

            # config = set_hyperparameter(full_event_test=True, time_window=item)
            if test_model == 0:
                model = 'hawkes_rnn_autoencoder_true'
                new_graph = tf.Graph()
                with new_graph.as_default():
                    with tf.device('/device:GPU:0'):
                        hawkes_rnn_test(config, cell_type=cell_type, autoencoder=autoencoder, model=model,
                                        save_last_hidden_state=save_last_hidden_state)
            elif test_model == 1:
                new_graph = tf.Graph()
                model = 'hawkes_rnn_autoencoder_false'
                with new_graph.as_default():
                    with tf.device('/device:GPU:0'):
                        hawkes_rnn_test(config, cell_type=cell_type,  autoencoder=-1, model=model,
                                        save_last_hidden_state=save_last_hidden_state)
            elif test_model == 2:
                new_graph = tf.Graph()
                model = 'vanilla_rnn_autoencoder_true'
                with new_graph.as_default():
                    with tf.device('/device:GPU:0'):
                        vanilla_rnn_test(config, cell_type=cell_type, autoencoder=autoencoder, model=model,
                                         save_last_hidden_state=save_last_hidden_state)
            elif test_model == 3:
                new_graph = tf.Graph()
                model = 'vanilla_rnn_autoencoder_false'
                with new_graph.as_default():
                    with tf.device('/device:GPU:0'):
                        vanilla_rnn_test(config, cell_type=cell_type, autoencoder=-1, model=model,
                                         save_last_hidden_state=save_last_hidden_state)
            elif test_model == 4:
                new_graph = tf.Graph()
                model = 'concat_hawkes_rnn_autoencoder_true'
                with new_graph.as_default():
                    with tf.device('/device:GPU:0'):
                        concat_hawkes_rnn_test(config, cell_type=cell_type, autoencoder=autoencoder, model=model,
                                               save_last_hidden_state=save_last_hidden_state)
            else:
                raise ValueError('invalid test model')


def hyperparameter_search(event_list, time_window):
    # 定义max sequence length为10为，3月心功能2级和1年心功能2级作为不同时间窗口的代表性事件，以Hawkes RNN Fuse为样本
    # 以贝叶斯优化的方式，0进行超参数搜索
    # 根据Cell Selection的结果，使用GRU作为Cell
    # 此处的贝叶斯策略仅仅是做toy用，具体测试结果仅做参考
    def tunning_wrapper(x):
        dae_weight = x[0][0]
        keep_rate_input = x[0][1]
        keep_rate_hidden = x[0][2]
        autoencoder = x[0][3]
        num_hidden = x[0][4]
        learning_rate = x[0][5]
        batch_size = x[0][6]
        config = set_hyperparameter(time_window=time_window, batch_size=batch_size, event_list=event_list,
                                    max_sequence_length=max_sequence_length, learning_rate=learning_rate,
                                    num_hidden=num_hidden, keep_rate_input=keep_rate_input,
                                    keep_rate_hidden=keep_rate_hidden, dae_weight=dae_weight)
        model = 'hyperparameter_search_hawkes_rnn_autoencoder_true'
        new_graph = tf.Graph()
        with new_graph.as_default():
            mean_auc = hawkes_rnn_test(config, cell_type='gru', autoencoder=autoencoder, model=model,
                                       save_last_hidden_state=False)
        return [[mean_auc]]

    max_sequence_length = 10
    bounds = [{'name': 'dae_weight', 'type': 'continuous', 'domain': (10**-1, 1)},
              {'name': 'keep_rate_input', 'type': 'continuous', 'domain': (0.8, 1)},
              {'name': 'keep_rate_hidden', 'type': 'continuous', 'domain': (0.8, 1)},
              {'name': 'autoencoder', 'type': 'discrete', 'domain': [i for i in range(16, 129)]},
              {'name': 'num_hidden', 'type': 'discrete', 'domain': [i for i in range(16, 129)]},
              {'name': 'learning_rate', 'type': 'continuous', 'domain': (0.0001, 0.1)},
              {'name': 'batch_size', 'type': 'discrete', 'domain': [i for i in range(64, 512)]}]

    opt_object = GPyOpt.methods.BayesianOptimization(tunning_wrapper, bounds, initial_design_numdata=1)

    opt_object.run_optimization()
    print('iter {}, optimum fx: {}, optimum x: {}'.format(0, opt_object.fx_opt, opt_object.x_opt))


def cell_search():
    # 以随机搜索方式，用fuse hawkes rnn在3次入院长度，一年心功能2期这个任务让运行10次
    # 每次进行超参数设计后都让LSTM和GRU各跑一次，综合比较，选择模型
    batch_size_list = [64, 128, 256, 512]
    num_hidden_list = [16, 32, 64, 128]
    autoencoder_list = [16, 32, 64, 128]

    time_window_list = ['三月', '一年']
    for item in time_window_list:
        for i in range(10):
            max_sequence_length = 3
            batch_size = batch_size_list[random.randint(0, 3)]
            learning_rate = 10**(random.uniform(-4, -1))
            num_hidden = num_hidden_list[random.randint(0, 3)]
            autoencoder = autoencoder_list[random.randint(0, 3)]
            keep_rate_hidden = random.uniform(0.8, 1)
            keep_rate_input = random.uniform(0.8, 1)
            dae_weight = 10**(random.uniform(-1, 0))

            config = set_hyperparameter(time_window=item, batch_size=batch_size,
                                        max_sequence_length=max_sequence_length, learning_rate=learning_rate,
                                        num_hidden=num_hidden, keep_rate_input=keep_rate_input,
                                        keep_rate_hidden=keep_rate_hidden, dae_weight=dae_weight)
            model = 'cell_search'
            new_graph = tf.Graph()
            with new_graph.as_default():
                hawkes_rnn_test(config, cell_type='lstm', autoencoder=autoencoder, model=model,
                                save_last_hidden_state=False)
            new_graph = tf.Graph()
            with new_graph.as_default():
                hawkes_rnn_test(config, cell_type='gru', autoencoder=autoencoder, model=model,
                                save_last_hidden_state=False)


if __name__ == '__main__':
    # cell_search()
    # hyperparameter_search(event_list=['心功能2级'], time_window='三月')
    performance_test(data_length=10, test_model=4, event_list=['其它'], save_last_hidden_state=True)
    # length_test(test_model=0, event_list=['其它'])
