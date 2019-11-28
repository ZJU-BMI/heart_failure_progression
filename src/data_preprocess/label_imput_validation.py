# encoding=utf-8
import random
import csv
import os
from itertools import islice
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
# 由于标签存在丢失现象，因此需要进行回归填补
# 此处的核心问题是处理数据的不平衡问题
# 按照计算，把丢失标签的数据进行重新分类（分为心功能1,2,3,4），若训练集和测试集服从同分布
# 分类器没有任何分类能力（指按照标签的比例，随机给新样本赋予标签），则四分类的准确率应当为40%
# 若使用SVM，调过类型权重后，可以使得模型在有能力进行少数类分类的前提下，验证集达到0.5左右的准确率，显著优于LR，与RF类似
# MLP大约可以达到更好的性能（0.55）左右，但是这两种模型受数据不平衡影响大，基本没有少数类型的分类能力，因此MLP未纳入考虑
# Sklearn没有给MLP定义类型权重的参数，理论上讲在Tensorflow框架下，通过修改损失函数，能够使得MLP也可以具备少数类分类的能力
# 但是这样做过于麻烦，因此直接按照SVM进行分类了


def main():
    source_data_path = os.path.abspath('..\\..\\resource\\预处理中间结果.csv')
    train_feature, train_label, valid_feature, valid_label, test_feature = get_data(source_data_path)

    class_weight = {1: 4.1, 2: 1, 3: 2.7, 4: 4}
    """
    clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial', max_iter=800,
                             class_weight=class_weight)
    clf.fit(train_feature, train_label)
    valid_prediction = clf.predict(valid_feature)
    train_prediction = clf.predict(train_feature)
    test_performance(valid_prediction, train_prediction, train_label, valid_label, 'LR')
    """
    clf = LinearSVC(tol=1e-4, max_iter=1000, class_weight=class_weight, C=0.1)
    clf.fit(train_feature, train_label)
    valid_prediction = clf.predict(valid_feature)
    train_prediction = clf.predict(train_feature)
    test_performance(valid_prediction, train_prediction, train_label, valid_label, 'SVM')

    # 单个MLP结果
    train_prediction, valid_prediction = mlp_classifier(train_feature, train_label, valid_feature)
    test_performance(valid_prediction, train_prediction, train_label, valid_label, 'MLP')

    # 集成学习
    sub_classifier_num = 5
    train_ensemble = list()
    valid_ensemble = list()
    for i in range(sub_classifier_num):
        train_prediction, valid_prediction = mlp_classifier(train_feature, train_label, valid_feature)
        train_ensemble.append(train_prediction)
        valid_ensemble.append(valid_prediction)
    train_ensemble = np.array(train_ensemble)
    valid_ensemble = np.array(valid_ensemble)

    train_ensemble_prediction = np.zeros([len(train_ensemble[0]), 4])
    valid_ensemble_prediction = np.zeros([len(valid_ensemble[0]), 4])
    for i in range(sub_classifier_num):
        for j in range(len(train_ensemble[i])):
            train_ensemble_prediction[j, int(train_ensemble[i][j])-1] += 1
        for j in range(len(valid_ensemble[i])):
            valid_ensemble_prediction[j, int(valid_ensemble[i][j])-1] += 1
    train_prediction = np.argmax(train_ensemble_prediction, axis=1) + 1
    valid_prediction = np.argmax(valid_ensemble_prediction, axis=1) + 1
    test_performance(valid_prediction, train_prediction, train_label, valid_label, 'Ensemble')


def mlp_classifier(train_feature, train_label, valid_feature):
    hidden_num_candidate = [3, 5, 7, 9]
    alpha = 10**random.uniform(-4, -2)
    learning_rate = 10**random.uniform(-4, -2)
    max_iter = random.randint(100, 200)
    momentum = 1 - 10**random.uniform(-2, -1)
    hidden_num = (hidden_num_candidate[random.randint(0, 3)], )

    clf = MLPClassifier(activation='relu', alpha=alpha, batch_size='auto', beta_1=0.9, beta_2=0.999,
                        early_stopping=False, epsilon=1e-08, hidden_layer_sizes=hidden_num, learning_rate='constant',
                        learning_rate_init=learning_rate, max_iter=max_iter, momentum=momentum, n_iter_no_change=10,
                        nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True, solver='lbfgs', tol=0.0001,
                        validation_fraction=0.1, verbose=False, warm_start=False)
    clf.fit(train_feature, train_label)
    valid_prediction = clf.predict(valid_feature)
    train_prediction = clf.predict(train_feature)
    return train_prediction, valid_prediction


def test_performance(valid_prediction, train_prediction, train_label, valid_label, name):
    # train
    train_count = 0
    for i in range(len(train_prediction)):
        pred = train_prediction[i]
        label = train_label[i]
        if pred == label:
            train_count += 1
    train_acc = train_count / len(train_prediction)
    print('{}训练集精度为{}'.format(name, train_acc))

    # valid
    valid_count = 0

    for i in range(len(valid_prediction)):
        pred = valid_prediction[i]
        label = valid_label[i]
        if pred == label:
            valid_count += 1
    valid_acc = valid_count / len(valid_prediction)
    print('{}验证集精度为{}'.format(name, valid_acc))

    one = 0
    two = 0
    three = 0
    four = 0
    for i in range(len(valid_prediction)):
        if valid_prediction[i] == 1:
            one += 1
        if valid_prediction[i] == 2:
            two += 1
        if valid_prediction[i] == 3:
            three += 1
        if valid_prediction[i] == 4:
            four += 1
    print('class 1: {}'.format(one))
    print('class 2: {}'.format(two))
    print('class 3: {}'.format(three))
    print('class 4: {}'.format(four))


def get_data(path):
    train_set = list()
    test_feature = list()

    with open(path, 'r', encoding='gbk', newline='') as file:
        csv_reader = csv.reader(file)
        for line in islice(csv_reader, 2, None):
            # 若一个数据为心源性入院且标签丢失，编入测试集；反之编入训练（验证）集
            # line[16] 之后是特征， line[5:9]是心功能的四个分级
            # 注意，若之前的数据预处理代码修改导致相应的index意义发生变化，此处要及时修改
            if line[15] == '1' and line[10] == '1':
                test_feature.append(line[16:])
                continue
            elif line[5] == '1' or line[6] == '1' or line[7] == '1' or line[8] == '1':
                # 至少存在心功能1,2,3,4级的标签
                event = line[5:9]
                train_set.append([event, line[16:]])
    random.shuffle(train_set)

    train_feature = list()
    train_label = list()
    valid_feature = list()
    valid_label = list()
    for i in range(len(train_set)):
        if i <= len(train_set)//3:
            valid_feature.append(train_set[i][1])
            event_flag = False
            for j in range(len(train_set[i][0])):
                if int(train_set[i][0][j]) == 1:
                    valid_label.append(j+1)
                    event_flag = True
            if not event_flag:
                valid_label.append(len(train_set[i][0]))

        else:
            train_feature.append(train_set[i][1])
            event_flag = False
            for j in range(len(train_set[i][0])):
                if int(train_set[i][0][j]) == 1:
                    train_label.append(j+1)
                    event_flag = True
            if not event_flag:
                train_label.append(len(train_set[i][0]))

    train_feature = np.array(train_feature, dtype=np.float)
    train_label = np.array(train_label, dtype=np.float)
    valid_feature = np.array(valid_feature, dtype=np.float)
    valid_label = np.array(valid_label, dtype=np.float)
    test_feature = np.array(test_feature, dtype=np.float)

    train_label_one = 0
    train_label_two = 0
    train_label_three = 0
    train_label_four = 0
    valid_label_one = 0
    valid_label_two = 0
    valid_label_three = 0
    valid_label_four = 0
    for item in train_label:
        if item == 1:
            train_label_one += 1
        if item == 2:
            train_label_two += 1
        if item == 3:
            train_label_three += 1
        if item == 4:
            train_label_four += 1
    for item in valid_label:
        if item == 1:
            valid_label_one += 1
        if item == 2:
            valid_label_two += 1
        if item == 3:
            valid_label_three += 1
        if item == 4:
            valid_label_four += 1
    print('train label class 1: {}'.format(train_label_one))
    print('train label class 2: {}'.format(train_label_two))
    print('train label class 3: {}'.format(train_label_three))
    print('train label class 4: {}'.format(train_label_four))

    print('valid label class 1: {}'.format(valid_label_one))
    print('valid label class 2: {}'.format(valid_label_two))
    print('valid label class 3: {}'.format(valid_label_three))
    print('valid label class 4: {}'.format(valid_label_four))
    return train_feature, train_label, valid_feature, valid_label, test_feature


if __name__ == '__main__':
    main()
