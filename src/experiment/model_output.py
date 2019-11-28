import tensorflow as tf
from sklearn import metrics
import numpy as np
from read_data import DataSource
from dynamic_fused_hawkes_rnn import fused_hawkes_rnn_model
from dynamic_concat_hawkes_rnn import concat_hawkes_model
from rnn_cell import RawCell, LSTMCell, GRUCell
import os


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


def fused_hawkes_rnn_test(shared_hyperparameter, cell_type, autoencoder, model, task):
    # test case = 0 五折交叉验证
    # test case = 1 tsne 用
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
        input_length = autoencoder + event_num
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
            fused_hawkes_rnn_model(num_steps=max_length, num_hidden=num_hidden, cell=cell, dae_weight=dae_weight,
                                   keep_rate_input=keep_rate_input, phase_indicator=phase_indicator,
                                   num_context=context_num, num_event=event_num, autoencoder_length=autoencoder)
        train_node = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

        node_list = [loss, prediction, event_placeholder, context_placeholder, y_placeholder, batch_size,
                     phase_indicator,  base_intensity, mutual_intensity, time_list, task_index, sequence_length,
                     final_state, train_node]

        validation_test(shared_hyperparameter, node_list, model_type='hawkes_rnn', model_name=model, task=task)


def concat_hawkes_rnn_test(shared_hyperparameter, cell_type, autoencoder, model, task):
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

        validation_test(shared_hyperparameter, node_list, model_type='hawkes_rnn', model_name=model, task=task)


def run_and_save_graph(data_source, max_step, node_list, validate_step_interval, task_name, experiment_config, model_type,
                       model_name, terminate_number=10):

    best_result = {'auc': 0, 'acc': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'prediction_label': list()}

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
    sess = tf.Session(config=config)
    sess.run(init)
    no_improve_count = 0
    minimum_loss = 10000

    save_folder = os.path.join(experiment_config['prediction_result_folder'], model_name)
    save_folder = os.path.join(save_folder, task_name)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    builder = save_model(sess, node_list, save_folder, model_version='1.14.0')

    for step in range(max_step):
        train_node = node_list[-1]
        loss = node_list[0]
        prediction = node_list[1]
        train_feed_dict, train_label = feed_dict_and_label(data_object=data_source, node_list=node_list,
                                                           base_intensity_value=base_intensity_value,
                                                           task_index=task_index, phase=2, model=model_type,
                                                           mutual_intensity_value=mutual_intensity_value,
                                                           task_name=task_name)
        sess.run(train_node, feed_dict=train_feed_dict)

        # Test Performance
        if step % validate_step_interval == 0:
            validate_feed_dict, validate_label =\
                feed_dict_and_label(data_object=data_source, node_list=node_list, task_index=task_index,
                                    base_intensity_value=base_intensity_value, phase=3, model=model_type,
                                    mutual_intensity_value=mutual_intensity_value, task_name=task_name)
            test_feed_dict, test_label = \
                feed_dict_and_label(data_object=data_source, node_list=node_list, task_index=task_index,
                                    base_intensity_value=base_intensity_value, phase=1, model=model_type,
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
            # 因此也就不存在"偷看标签"的问题
            # 之所以这么设计，主要是因为标准的Early Stop模型要不断的存取模型副本，然后训练完了再计算测试集性能
            #  这个跑法有点麻烦，而且我们其实也不是特别需要把模型存起来

            #  性能记录与Early Stop的实现，前三次不记录（防止出现训练了还不如不训练的情况）
            if step < validate_step_interval * 3:
                continue

            if validate_loss >= minimum_loss:
                # 连续10次验证集性能无差别，则停止训练
                if no_improve_count >= validate_step_interval * terminate_number:
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

                builder.save()

            if validate_label.sum() < 0.5:
                break

    # 输出最优结果与当前模型（通过node_list输出）
    return best_result, node_list, sess


def save_model(sess, node_list, export_path_base, model_version='1.14.0'):
    # Export model
    # WARNING(break-tutorial-inline-code): The following code snippet is
    # in-lined in tutorials, please update tutorial documents accordingly
    # whenever code changes.
    export_path = os.path.join(
        tf.compat.as_bytes(export_path_base),
        tf.compat.as_bytes(str(model_version)))
    print('Exporting trained model to', export_path)
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)

    loss, prediction, event_placeholder, context_placeholder, y_placeholder, batch_size,\
        phase_indicator, base_intensity, mutual_intensity, time_list, task_index, sequence_length,\
            final_state, train_node = node_list
    event_sig = tf.saved_model.utils.build_tensor_info(event_placeholder)
    context_sig = tf.saved_model.utils.build_tensor_info(context_placeholder)
    pred_sig = tf.saved_model.utils.build_tensor_info(prediction)
    phase_sig = tf.saved_model.utils.build_tensor_info(phase_indicator)
    base_intensity_sign = tf.saved_model.utils.build_tensor_info(base_intensity)
    batch_size_sig = tf.saved_model.utils.build_tensor_info(batch_size)
    mutual_intensity_sig = tf.saved_model.utils.build_tensor_info(mutual_intensity)
    task_index_sig = tf.saved_model.utils.build_tensor_info(task_index)
    time_list_sig = tf.saved_model.utils.build_tensor_info(time_list)
    sequence_length_sig = tf.saved_model.utils.build_tensor_info(sequence_length)

    prediction_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs={'event': event_sig, 'context': context_sig, 'base': base_intensity_sign, 'batch': batch_size_sig,
                    'mutual': mutual_intensity_sig, 'phase': phase_sig, 'task': task_index_sig,
                    'time_list': time_list_sig, 'sequence_length': sequence_length_sig},
            outputs={'scores': pred_sig},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

    builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            'predict':
                prediction_signature,
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                prediction_signature,
        },
        main_op=tf.tables_initializer(),
        strip_default_attrs=True)

    return builder


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
    # 全数据测试
    elif phase == 4:
        input_event, input_context, input_t, input_sequence_length, input_y = data_object.get_all_data(task_name)
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
        _, _, event_placeholder, context_placeholder, y_placeholder, batch_size, phase_indicator, base_intensity, \
            mutual_intensity, time_list, task_index_, sequence_length, _, _ = node_list
        feed_dict = {event_placeholder: input_event, y_placeholder: input_y, batch_size: batch_size_value,
                     time_list: time_interval_value, phase_indicator: phase_value,
                     context_placeholder: input_context, mutual_intensity: mutual_intensity_value,
                     base_intensity: base_intensity_value, task_index_: task_index,
                     sequence_length: input_sequence_length}
        return feed_dict, input_y
    else:
        raise ValueError('invalid model name')


def validation_test(experiment_config, node_list, model_type, model_name, task):
    max_length = experiment_config['max_sequence_length']
    data_folder = experiment_config['data_folder']
    batch_size = experiment_config['batch_size']
    max_iter = experiment_config['max_iter']
    validate_step_interval = experiment_config['validate_step_interval']
    data_fraction = experiment_config['data_fraction']
    test_case = experiment_config['test_case']

    result_folder = os.path.join(experiment_config['prediction_result_folder'], test_case+'_'+model_name)

    # 从洗过的数据中读取数据
    data_source = DataSource(data_folder, data_length=max_length, validate_fold_num=0,
                             batch_size=batch_size, repeat=0, read_fraction=data_fraction)

    best_result, _, sess = run_and_save_graph(data_source=data_source, max_step=max_iter,  node_list=node_list,
                                              validate_step_interval=validate_step_interval, model_type=model_type,
                                              model_name=model_name,
                                              experiment_config=experiment_config, task_name=task)
    sess.close()



def set_hyperparameter(time_window, test_case, data_fraction=1.0, max_sequence_length=10, batch_size=256,
                       dae_weight=1.0, learning_rate=0.001, num_hidden=32, keep_rate_hidden=1, keep_rate_input=0.8,
                       event_list=None):
    """
    :return:
    """
    data_folder = os.path.abspath('../../resource/rnn_data')
    mutual_intensity_path = os.path.abspath('../../resource/hawkes_result/mutual.npy')
    base_intensity_path = os.path.abspath('../../resource/hawkes_result/base.npy')
    model_save_path = os.path.abspath('../../resource/model_cache')
    model_graph_save_path = os.path.abspath('../../resource/model_diagram')
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
                           '其它', '肺病', '肾病入院']
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
                            'test_case': test_case, 'prediction_result_folder': prediction_result_folder,
                            'model_path': model_save_path, 'model_graph_save_path': model_graph_save_path,
                            'data_fraction': data_fraction}
    return experiment_configure


def length_test(test_model, event_list, length_list):
    for i in length_list:
        performance_test(data_length=i, test_model=test_model, event_list=event_list, test_case='length_test')


def data_fraction_test(test_model, event_list):
    for item in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        performance_test(data_length=10, test_model=test_model, event_list=event_list, test_case='data_fraction_test',
                         data_fraction=item)


def performance_test(data_length, test_model, test_case, event_list=None, data_fraction=1.0):
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
                                        keep_rate_hidden=keep_rate_hidden, dae_weight=dae_weight,
                                        data_fraction=data_fraction, test_case=test_case)

            for task in config['event_list']:
                # config = set_hyperparameter(full_event_test=True, time_window=item)
                if test_model == 0:
                    model = 'fused_hawkes_rnn_autoencoder_true'
                    new_graph = tf.Graph()
                    with new_graph.as_default():
                        with tf.device('/device:GPU:0'):
                            fused_hawkes_rnn_test(config, cell_type=cell_type, autoencoder=autoencoder, model=model,
                                                  task=task)
                elif test_model == 1:
                    new_graph = tf.Graph()
                    model = 'fused_hawkes_rnn_autoencoder_false'
                    with new_graph.as_default():
                        with tf.device('/device:GPU:0'):
                            fused_hawkes_rnn_test(config, cell_type=cell_type,  autoencoder=-1, model=model, task=task)
                elif test_model == 4:
                    new_graph = tf.Graph()
                    model = 'concat_hawkes_rnn_autoencoder_true'
                    with new_graph.as_default():
                        with tf.device('/device:GPU:0'):
                            concat_hawkes_rnn_test(config, cell_type=cell_type, autoencoder=autoencoder, model=model,
                                                   task=task)
                elif test_model == 5:
                    new_graph = tf.Graph()
                    model = 'concat_hawkes_rnn_autoencoder_false'
                    with new_graph.as_default():
                        with tf.device('/device:GPU:0'):
                            concat_hawkes_rnn_test(config, cell_type=cell_type, autoencoder=-1, model=model,
                                                   task=task)
                else:
                    raise ValueError('invalid test model')


if __name__ == '__main__':
    # cell_search('lstm')
    # hyperparameter_search(event_list=['心功能2级'], time_window='三月')
    performance_test(data_length=10, test_model=0, event_list=None, test_case='performance_test')
    # data_fraction_test(test_model=5, event_list=None)
    print('complete')
