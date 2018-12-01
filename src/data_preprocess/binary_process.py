import csv
import os
import re
import datetime
from itertools import islice


def main():
    data_path = os.path.abspath('..\\..\\resource\\未预处理长期纵向数据.csv')
    save_data_path = os.path.abspath('..\\..\\resource\\二值化的长期纵向数据.csv')
    data_dict, event_list, contextual_set = read_data(data_path, event_count=11)
    binary_data_dict, revised_contextual_set = binary(data_dict, contextual_set)

    # 写数据
    data_to_write = list()
    head = list()
    head.append('patient_id')
    head.append('visit_id')
    for item in event_list:
        head.append(item)
    for item in revised_contextual_set:
        head.append(item)
    data_to_write.append(head)

    general_list = head[2:]

    for patient_id in binary_data_dict:
        # 强制有序输出
        for i in range(100):
            if binary_data_dict[patient_id].__contains__(str(i)):
                visit_id = str(i)
                row = [patient_id, visit_id]
                for key in general_list:
                    value = binary_data_dict[patient_id][visit_id][key]
                    row.append(value)
                data_to_write.append(row)
    with open(save_data_path, 'w', encoding='gbk', newline='') as file:
        csv.writer(file).writerows(data_to_write)


def data_regenerate(event_dict, origin_file, write_path):
    """
    读取既有数据，然后按照event_dict的内容填补缺失数据，然后删除数据标签丢失这一项目
    :param event_dict:
    :param write_path:
    :param origin_file:
    :return:
    """
    data_dict = dict()
    feature_dict = dict()
    with open(origin_file, 'r', encoding='gbk', newline='') as file:
        csv_reader = csv.reader(file)
        head_flag = True
        skip_second_line = True
        for line in csv_reader:
            if head_flag:
                for i in range(2, len(line)):
                    feature_dict[i] = line[i]
                head_flag = False
                continue

            # 跳过第二行（丢失率说明）
            if skip_second_line:
                skip_second_line = False
                continue

            patient_id = line[0]
            visit_id = line[1]

            if not data_dict.__contains__(patient_id):
                data_dict[patient_id] = dict()
            data_dict[patient_id][visit_id] = dict()

            for i in range(2, len(line)):
                data_dict[patient_id][visit_id][feature_dict[i]] = line[i]

    # 事件替换
    for patient_id in event_dict:
        for visit_id in event_dict[patient_id]:
            event = event_dict[patient_id][visit_id]
            data_dict[patient_id][visit_id][event] = 1
    for patient_id in data_dict:
        for visit_id in data_dict[patient_id]:
            data_dict[patient_id][visit_id].pop('心功能标签丢失')

    # 写数据
    general_list = list()
    for patient_id in data_dict:
        for visit_id in data_dict[patient_id]:
            for key in data_dict[patient_id][visit_id]:
                general_list.append(key)
            break
        break
    data_to_write = list()
    head = list()
    head.append('patient_id')
    head.append('visit_id')
    for item in general_list:
        head.append(item)
    data_to_write.append(head)
    for patient_id in data_dict:
        # 强制有序输出
        for i in range(100):
            if data_dict[patient_id].__contains__(str(i)):
                visit_id = str(i)
                row = [patient_id, visit_id]
                for key in general_list:
                    value = data_dict[patient_id][visit_id][key]
                    row.append(value)
                data_to_write.append(row)
    with open(write_path, 'w', encoding='gbk', newline='') as file:
        csv.writer(file).writerows(data_to_write)

    return data_dict


def binary(data_dict, contextual_set):
    """
    此处的Binary策略为定制的，主要是这两天实在写不动代码了
    以后可以修复为有config配置的二值化策略
    :return:
    """
    for patient_id in data_dict:
        for visit_id in data_dict[patient_id]:
            bp_dia = float(data_dict[patient_id][visit_id].pop('血压Low'))
            bp_sys = float(data_dict[patient_id][visit_id].pop('血压high'))
            height = float(data_dict[patient_id][visit_id].pop('身高'))
            weight = float(data_dict[patient_id][visit_id].pop('体重'))
            bp = float(data_dict[patient_id][visit_id].pop('脉搏'))
            egfr = float(data_dict[patient_id][visit_id].pop('EGFR'))
            age = float(data_dict[patient_id][visit_id].pop('年龄'))
            los = float(data_dict[patient_id][visit_id].pop('住院天数'))
            ef = float(data_dict[patient_id][visit_id].pop('射血分数'))

            contextual_set.discard('血压Low')
            contextual_set.discard('射血分数')
            contextual_set.discard('血压high')
            contextual_set.discard('身高')
            contextual_set.discard('体重')
            contextual_set.discard('脉搏')
            contextual_set.discard('EGFR')
            contextual_set.discard('年龄')
            contextual_set.discard('住院天数')

            if los <= 7:
                data_dict[patient_id][visit_id]['住院一周以内'] = 1
                data_dict[patient_id][visit_id]['住院半月以内'] = 0
                data_dict[patient_id][visit_id]['住院半月以上'] = 0
            elif 7 < los < 15:
                data_dict[patient_id][visit_id]['住院一周以内'] = 0
                data_dict[patient_id][visit_id]['住院半月以内'] = 1
                data_dict[patient_id][visit_id]['住院半月以上'] = 0
            else:
                data_dict[patient_id][visit_id]['住院一周以内'] = 0
                data_dict[patient_id][visit_id]['住院半月以内'] = 0
                data_dict[patient_id][visit_id]['住院半月以上'] = 1
            contextual_set.add('住院一周以内')
            contextual_set.add('住院半月以内')
            contextual_set.add('住院半月以上')

            if age < 40:
                data_dict[patient_id][visit_id]['青年患者'] = 1
                data_dict[patient_id][visit_id]['中年患者'] = 0
                data_dict[patient_id][visit_id]['老年患者'] = 0
            elif 40 <= age < 60:
                data_dict[patient_id][visit_id]['青年患者'] = 0
                data_dict[patient_id][visit_id]['中年患者'] = 1
                data_dict[patient_id][visit_id]['老年患者'] = 0
            else:
                data_dict[patient_id][visit_id]['青年患者'] = 0
                data_dict[patient_id][visit_id]['中年患者'] = 0
                data_dict[patient_id][visit_id]['老年患者'] = 1
            contextual_set.add('青年患者')
            contextual_set.add('中年患者')
            contextual_set.add('老年患者')

            if egfr >= 80:
                data_dict[patient_id][visit_id]['肾小球滤过率正常'] = 1
                data_dict[patient_id][visit_id]['肾小球滤过率偏低'] = 0
                data_dict[patient_id][visit_id]['肾小球滤过率严重偏低'] = 0
            elif 30 <= egfr < 80:
                data_dict[patient_id][visit_id]['肾小球滤过率正常'] = 0
                data_dict[patient_id][visit_id]['肾小球滤过率偏低'] = 1
                data_dict[patient_id][visit_id]['肾小球滤过率严重偏低'] = 0
            else:
                data_dict[patient_id][visit_id]['肾小球滤过率正常'] = 0
                data_dict[patient_id][visit_id]['肾小球滤过率偏低'] = 0
                data_dict[patient_id][visit_id]['肾小球滤过率严重偏低'] = 1
            contextual_set.add('肾小球滤过率正常')
            contextual_set.add('肾小球滤过率偏低')
            contextual_set.add('肾小球滤过率严重偏低')

            if bp > 100:
                data_dict[patient_id][visit_id]['脉搏心动过速'] = 1
            else:
                data_dict[patient_id][visit_id]['脉搏心动过速'] = 0
            contextual_set.add('脉搏心动过速')

            if ef > 50:
                data_dict[patient_id][visit_id]['射血分数异常'] = 1
            else:
                data_dict[patient_id][visit_id]['射血分数异常'] = 0
            contextual_set.add('射血分数异常')

            if bp_dia > 90:
                data_dict[patient_id][visit_id]['舒张压偏高'] = 1
            else:
                data_dict[patient_id][visit_id]['舒张压偏高'] = 0
            contextual_set.add('舒张压偏高')

            if bp_sys > 140:
                data_dict[patient_id][visit_id]['收缩压偏高'] = 1
            else:
                data_dict[patient_id][visit_id]['收缩压偏高'] = 0
            contextual_set.add('收缩压偏高')

            bmi = weight/((height/100)*(height/100))
            if bmi <= 18.5:
                data_dict[patient_id][visit_id]['BMI偏低'] = 1
                data_dict[patient_id][visit_id]['BMI正常'] = 0
                data_dict[patient_id][visit_id]['BMI超重'] = 0
                data_dict[patient_id][visit_id]['BMI肥胖'] = 0
            elif 18.5 < bmi <= 23.9:
                data_dict[patient_id][visit_id]['BMI偏低'] = 0
                data_dict[patient_id][visit_id]['BMI正常'] = 1
                data_dict[patient_id][visit_id]['BMI超重'] = 0
                data_dict[patient_id][visit_id]['BMI肥胖'] = 0
            elif 24 < bmi <= 27:
                data_dict[patient_id][visit_id]['BMI偏低'] = 0
                data_dict[patient_id][visit_id]['BMI正常'] = 0
                data_dict[patient_id][visit_id]['BMI超重'] = 1
                data_dict[patient_id][visit_id]['BMI肥胖'] = 0
            else:
                data_dict[patient_id][visit_id]['BMI偏低'] = 0
                data_dict[patient_id][visit_id]['BMI正常'] = 0
                data_dict[patient_id][visit_id]['BMI超重'] = 0
                data_dict[patient_id][visit_id]['BMI肥胖'] = 1
            contextual_set.add('BMI偏低')
            contextual_set.add('BMI正常')
            contextual_set.add('BMI超重')
            contextual_set.add('BMI肥胖')

    return data_dict, contextual_set


def read_data(file_path, event_count=11):
    data_dict = dict()
    feature_dict = dict()
    event_list = list()
    contextual_set = set()
    with open(file_path, 'r', encoding='gbk', newline='') as file:
        csv_reader = csv.reader(file)
        head_flag = True
        for line in csv_reader:
            if head_flag:
                for i in range(2, len(line)):
                    feature_dict[i] = line[i]
                    if i <= event_count+3:
                        # 确保前14项包含了所有事件及时间差，后面的特征项的顺序可以不管
                        content = line[i]
                        event_list.append(content)
                    else:
                        content = line[i]
                        contextual_set.add(content)
                head_flag = False
                continue

            patient_id = line[0]
            visit_id = line[1]

            if not data_dict.__contains__(patient_id):
                data_dict[patient_id] = dict()
            data_dict[patient_id][visit_id] = dict()

            for i in range(2, len(line)):
                data_dict[patient_id][visit_id][feature_dict[i]] = line[i]
    return data_dict, event_list, contextual_set


if __name__ == '__main__':
    main()
