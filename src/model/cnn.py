import numpy as np
import tensorflow as tf
import os
from sklearn import metrics
import autoencoder
import datetime
from read_data import DataSource
import csv


def cnn_model(num_steps, num_context, num_event, keep_rate_input, dae_weight, phase_indicator,
              autoencoder_length, filter_num, filter_size, autoencoder_initializer):
    with tf.name_scope('data_source'):
        # 标准输入规定为TBD
        batch_size = tf.placeholder(tf.int32, [], name='batch_size')
        event_placeholder = tf.placeholder(tf.float32, [num_steps, None, num_event], name='event_placeholder')
        context_placeholder = tf.placeholder(tf.float32, [num_steps, None, num_context], name='context_placeholder')
        time_list = tf.placeholder(tf.int32, [None, num_steps], name='time_list')
        y_placeholder = tf.placeholder(tf.float32, [None, 1], name='y_placeholder')

    with tf.name_scope('autoencoder'):
        # input_x 用于计算重构原始向量时产生的误差
        processed_input, autoencoder_weight = autoencoder.denoising_autoencoder(
            phase_indicator, context_placeholder, keep_rate_input, autoencoder_length, autoencoder_initializer)

    with tf.name_scope('cnn'):
        predictions, scores = _cnn_detail(processed_input=processed_input, event_list=event_placeholder,
                                          time_list=time_list, filter_num=filter_num, filter_size_list=filter_size,
                                          scope='cnn')

    with tf.name_scope('loss'):
        with tf.name_scope('pred_loss'):
            loss_pred = tf.losses.sigmoid_cross_entropy(logits=scores,
                                                        multi_class_labels=y_placeholder)

        with tf.name_scope('dae_loss'):
            if autoencoder_length > 0:
                loss_dae = autoencoder.autoencoder_loss(embedding=processed_input, origin_input=context_placeholder,
                                                        weight=autoencoder_weight)
            else:
                loss_dae = 0

        with tf.name_scope('loss_sum'):
            loss = loss_pred + loss_dae * dae_weight

    return loss, predictions, event_placeholder, context_placeholder, y_placeholder, batch_size, phase_indicator, \
        time_list


def _cnn_detail(processed_input, event_list, filter_num, filter_size_list, time_list, scope, num_classes=1,
                drop_keep_prob=0.8):
    # concat
    time_list_expand = tf.cast(tf.expand_dims(tf.transpose(time_list), axis=-1), tf.float32)
    feature_input = tf.concat([event_list, processed_input, time_list_expand], axis=2)
    feature_input_expand = tf.expand_dims(feature_input, -1)
    embedding_size = int(feature_input.shape[2])
    sequence_length = int(feature_input.shape[0])
    feature_input_expand = tf.transpose(feature_input_expand, [1, 0, 2, 3])
    pooled_outputs = []
    for i, filter_size in enumerate(filter_size_list):
        with tf.name_scope("{}_conv-maxpool-{}".format(scope, filter_size)):
            # Convolution Layer
            filter_shape = [filter_size, embedding_size, 1, filter_num]
            w = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[filter_num]), name="b")
            conv = tf.nn.conv2d(
                feature_input_expand,
                w,
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="conv")
            # Apply nonlinearity
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            # Maxpooling over the outputs
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, sequence_length - filter_size + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool")
            pooled_outputs.append(pooled)
    # Combine all the pooled features
    num_filters_total = filter_num * len(filter_size_list)
    h_pool = tf.concat(pooled_outputs, 3)
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

    # Add dropout
    with tf.name_scope("dropout"):
        h_drop = tf.nn.dropout(h_pool_flat, drop_keep_prob)

    # Final (unnormalized) scores and predictions
    with tf.name_scope("output"):
        w = tf.get_variable(
            "W",
            shape=[num_filters_total, num_classes],
            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
        scores = tf.nn.xw_plus_b(h_drop, w, b, name="scores")
        predictions = tf.argmax(scores, 1, name="predictions")

    return predictions, scores


def cnn_test(max_length, context_num, event_num, keep_rate_input, dae_weight, autoencoder_length, filter_num,
             filter_size, learning_rate, data_folder, data_fraction, event_list, max_iter,
             test_case, validate_step_interval, model):
    parameter_list = [max_length, context_num, event_num, keep_rate_input, dae_weight, autoencoder_length, filter_num,
                      filter_size, learning_rate, model]

    g = tf.Graph()
    with g.as_default():
        with tf.name_scope('phase'):
            phase_indicator = tf.placeholder(tf.int32, shape=[], name="phase_indicator")

        loss, predictions, event_placeholder, context_placeholder, y_placeholder, batch_size, phase_indicator, \
            time_list = cnn_model(num_steps=max_length, num_context=context_num, num_event=event_num,
                                  keep_rate_input=keep_rate_input, dae_weight=dae_weight,  filter_num=filter_num,
                                  phase_indicator=phase_indicator, autoencoder_length=autoencoder_length,
                                  filter_size=filter_size, autoencoder_initializer=tf.initializers.orthogonal())
        train_node = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

        node_list = [loss, predictions, event_placeholder, context_placeholder, y_placeholder, batch_size,
                     phase_indicator, time_list, train_node]

        mean_auc = five_fold_validation_default(parameter_list, node_list, model_type='cnn', model_name=model,
                                                batch_size=batch_size, data_folder=data_folder, event_list=event_list,
                                                data_fraction=data_fraction, max_iteration=max_iter,
                                                test_case=test_case, validate_step_interval=validate_step_interval)
    return mean_auc


def five_fold_validation_default(parameter_list, node_list, model_type, model_name, event_list, batch_size,
                                 data_fraction, data_folder, test_case, max_iteration, validate_step_interval):
    max_length, context_num, event_num, keep_rate_input, dae_weight, autoencoder_length, filter_num, filter_size, \
        learning_rate, model = parameter_list

    result_folder = os.path.join(os.path.abspath('../../resource/prediction_result/'), test_case+'_'+model_name)

    # 五折交叉验证，跑十次
    best_auc = dict()
    for task in event_list:
        result_record = dict()
        prediction_label_dict = dict()
        for j in range(5):
            if not result_record.__contains__(j):
                result_record[j] = dict()
                prediction_label_dict[j] = dict()
            for i in range(10):

                # 从洗过的数据中读取数据
                data_source = DataSource(data_folder, data_length=max_length, validate_fold_num=j,
                                         batch_size=batch_size, repeat=i, read_fraction=data_fraction)

                best_result, _, sess = run_graph(data_source=data_source, max_step=max_iteration,  node_list=node_list,
                                                 validate_step_interval=validate_step_interval, model=model_type,
                                                 task_name=task)
                sess.close()
                prediction_label = best_result.pop('prediction_label')
                result_record[j][i] = best_result
                prediction_label_dict[j][i] = prediction_label
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
        save_result(result_folder, result_record, prediction_label_dict, task)

    # 最后返回综合AUC作为性能评价指标（其实不太合适，但是就这样吧）
    mean_auc = 0
    for key in best_auc:
        mean_auc += best_auc[key]
    mean_auc = mean_auc/len(best_auc)
    return mean_auc


def run_graph(data_source, max_step, node_list, validate_step_interval, task_name, model, terminate_number=10):

    best_result = {'auc': 0, 'acc': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'prediction_label': list()}

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allow_growth = True

    init = tf.global_variables_initializer()
    sess = tf.Session(config=config)
    sess.run(init)
    no_improve_count = 0
    minimum_loss = 10000

    for step in range(max_step):
        train_node = node_list[-1]
        loss = node_list[0]
        prediction = node_list[1]
        train_feed_dict, train_label = feed_dict_and_label(data_object=data_source, node_list=node_list,
                                                           phase=2, model=model, task_name=task_name)
        sess.run(train_node, feed_dict=train_feed_dict)

        # Test Performance
        if step % validate_step_interval == 0:
            validate_feed_dict, validate_label =\
                feed_dict_and_label(data_object=data_source, node_list=node_list, phase=3, model=model,
                                    task_name=task_name)
            test_feed_dict, test_label = \
                feed_dict_and_label(data_object=data_source, node_list=node_list, phase=1, model=model,
                                    task_name=task_name)

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

            if validate_label.sum() < 0.5:
                break

    # 输出最优结果与当前模型（通过node_list输出）
    return best_result, node_list, sess


def feed_dict_and_label(data_object, node_list, task_name, phase, model):
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

    if model == 'cnn':
        _, _, event_placeholder, context_placeholder, y_placeholder, batch_size, phase_indicator, sequence_length, _,\
            _ = node_list
        feed_dict = {event_placeholder: input_event, y_placeholder: input_y, batch_size: batch_size_value,
                     phase_indicator: phase_value, context_placeholder: input_context,
                     sequence_length: input_sequence_length}
        return feed_dict, input_y
    else:
        raise ValueError('invalid model name')


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


def save_result(save_folder, result_record, prediction_label_dict, task_name):

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    current_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

    # 存储性能
    max_sequence_length = experiment_config['max_sequence_length']
    data_fraction = experiment_config['data_fraction']

    # 存储预测与标签
    save_path = os.path.join(save_folder,
                             'length_{}_prediction_label_{}_{}_data_fraction_{}.csv'.
                             format(max_sequence_length, task_name, current_time, data_fraction))
    data_to_write = [['label', 'prediction']]
    for key in experiment_config:
        data_to_write.append([key, experiment_config[key]])
    for j in prediction_label_dict:
        for i in prediction_label_dict[j]:
            for k in range(len(prediction_label_dict[j][i])):
                data_to_write.append(prediction_label_dict[j][i][k])
    with open(save_path, 'w', encoding='gbk', newline='') as file:
        csv.writer(file).writerows(data_to_write)

    # 存储综合性能
    save_path = os.path.join(save_folder, 'result_length_{}_{}_{}.csv'.format(max_sequence_length,
                                                                              task_name, current_time))
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


def performance_test(data_length, test_model, test_case, event_list=None, data_fraction=1.0):
    max_iter = 10000
    validate_step_interval = 20
    context_num = 189
    event_num = 11

    data_folder = os.path.abspath('../../resource/rnn_data')

    max_length = data_length
    batch_size = 256
    learning_rate = 0.001
    # dae的keep prob
    keep_rate_input = 0.8
    dae_weight = 1
    autoencoder_length = 16
    filter_num = 2
    filter_sizes = [2, 2]

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

    time_window_list = ['一年', '三月']
    for item in time_window_list:
        # config = set_hyperparameter(full_event_test=True, time_window=item)
        new_graph = tf.Graph()
        with new_graph.as_default():
            with tf.device('/device:GPU:0'):
                cnn_test(max_length, context_num, event_num, keep_rate_input, dae_weight, autoencoder_length,
                         filter_num, filter_sizes, learning_rate, data_folder, data_fraction, event_list, max_iter,
                         test_case, validate_step_interval, test_model)


if __name__ == '__main__':
    performance_test()



