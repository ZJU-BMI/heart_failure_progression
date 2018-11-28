# encoding=utf-8-sig
import os
import numpy as np
import random


class DataSource(object):
    def __init__(self, data_folder, data_length, repeat, test_fold_num, batch_size, event_count=11):
        """
        :param data_folder:
        :param data_length:
        :param test_fold_num:
        :param repeat:
        :param batch_size:
        :param event_count:
        """
        self.__data_folder = data_folder
        self.__test_fold = test_fold_num
        self.__batch_size = batch_size
        self.__data_length = data_length
        self.__repeat = repeat
        self.__event_count = event_count

        train_feature_list, train_label_dict, test_feature_list, test_label_dict = self.__read_data()
        if batch_size >= len(train_feature_list):
            raise ValueError("Batch Size Too Large")
        self.__test_feature = test_feature_list
        self.__test_label = test_label_dict
        self.__raw_train_feature = train_feature_list
        self.__raw_train_label = train_label_dict
        self.__shuffled_train_feature, self.__shuffled_train_label = self.__shuffle_data()

        self.__current_batch_index = 0

    def get_test_label(self):
        return self.__test_label

    def get_test_feature(self):
        test_feature = self.__test_feature
        event_count = self.__event_count
        batch_time = test_feature[:, :, event_count]
        batch_feature = np.concatenate([test_feature[:, :, 0: event_count],
                                        test_feature[:, :, event_count+1:]], axis=2)
        return batch_feature, batch_time

    def get_next_batch(self, key_name):
        batch_size = self.__batch_size
        index = self.__current_batch_index
        batch_feature = self.__shuffled_train_feature[index*batch_size:(index+1)*batch_size]
        batch_label = self.__shuffled_train_label[key_name][index*batch_size:(index+1)*batch_size]
        batch_feature = np.array(batch_feature)
        batch_label = np.array(batch_label)
        batch_label = batch_label[:, np.newaxis]

        if batch_size*(index+2) > len(self.__shuffled_train_feature):
            self.__shuffled_train_feature, self.__shuffled_train_label = self.__shuffle_data()
            self.__current_batch_index = 0
        else:
            self.__current_batch_index += 1

        event_count = self.__event_count
        batch_time = batch_feature[:, :, event_count]
        batch_feature = np.concatenate([batch_feature[:, :, 0: event_count],
                                        batch_feature[:, :, event_count+1:]], axis=2)
        return batch_feature, batch_time, batch_label

    def __shuffle_data(self):
        # 随机化数据的要求是，随机化不会破坏原始数据中的特征、标签之间的对应关系
        shuffled_train_feature = list()
        shuffled_train_label = dict()

        # 构建随机化序列
        shuffle_index = [i for i in range(len(self.__raw_train_feature))]
        random.shuffle(shuffle_index)

        # 填充关键词
        for key in self.__raw_train_label:
            shuffled_train_label[key] = list()

        for i in shuffle_index:
            shuffled_train_feature.append(self.__raw_train_feature[i])
            for key in shuffled_train_label:
                shuffled_train_label[key].append(self.__raw_train_label[key][i])

        return shuffled_train_feature, shuffled_train_label

    def __read_data(self):
        data_length = self.__data_length
        data_folder = self.__data_folder
        repeat = self.__repeat
        train_feature_list = list()
        test_feature_list = None
        train_label_dict = dict()
        test_label_dict = dict()
        for i in range(5):
            feature_name = 'length_{}_repeat_{}_fold_{}_feature.npy'.format(str(data_length), str(repeat), str(i))
            feature = np.load(os.path.join(data_folder, feature_name))
            if i == self.__test_fold:
                test_feature_list = feature
                continue

            for j in range(len(feature)):
                train_feature_list.append(feature[j])
        train_feature_list = np.array(train_feature_list)

        # 构建event列表
        label_candidate = ['其它', '再血管化手术', '心功能1级', '心功能2级', '心功能3级', '心功能4级', '死亡', '癌症',
                           '糖尿病入院', '肺病', '肾病入院']
        time_candidate = ['三月', '半年', '一年', '两年']
        event_list = list()
        for item_1 in label_candidate:
            for item_2 in time_candidate:
                event_list.append(item_2+item_1)
        # 读取数据
        for key_name in event_list:
            if not train_label_dict.__contains__(key_name):
                train_label_dict[key_name] = list()
            if not test_label_dict.__contains__(key_name):
                test_label_dict[key_name] = list()
            for i in range(5):
                label_name = 'length_{}_repeat_{}_fold_{}_{}_label.npy'.format(
                    str(data_length), str(repeat), str(i), key_name)
                label = np.load(os.path.join(data_folder, label_name))
                if i == self.__test_fold:
                    for item in label:
                        test_label_dict[key_name].append(item)
                    test_label_dict[key_name] = np.array(test_label_dict[key_name])
                    continue

                for item in label:
                    train_label_dict[key_name].append(item)
            train_label_dict[key_name] = np.array(train_label_dict[key_name])

        return train_feature_list, train_label_dict, test_feature_list, test_label_dict


def unit_test():
    data_folder = os.path.abspath('..\\..\\resource\\rnn_data')
    data_length = 3
    test_fold_num = 0
    batch_size = 128
    data_source = DataSource(data_folder, data_length, 8, test_fold_num, batch_size)
    for i in range(100):
        batch_data = data_source.get_next_batch('一年肾病入院')
        print(batch_data[0])
        print(batch_data[1])


if __name__ == '__main__':
    unit_test()
