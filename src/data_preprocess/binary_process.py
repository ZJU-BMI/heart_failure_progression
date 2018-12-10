import csv
import os


def main():
    data_path = os.path.abspath('..\\..\\resource\\未预处理长期纵向数据.csv')
    binary_path = os.path.abspath('..\\..\\resource\\二值化策略.csv')
    save_data_path = os.path.abspath('..\\..\\resource\\二值化后的长期纵向数据.csv')
    data_dict, event_list, contextual_list = read_data(data_path, event_count=11)
    binary_process_dict = read_binary_process_feature(binary_path)
    binary_data_dict, revised_content_list = binary(data_dict, binary_process_dict, contextual_list)

    # 写数据
    data_to_write = list()
    head = list()
    head.append('patient_id')
    head.append('visit_id')
    for item in event_list:
        head.append(item)
    for item in revised_content_list:
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


def read_binary_process_feature(data_path):
    """
    目前定义二值化操作的feature为三类
    1. 根据原始数据即可二值化的，创造3维哑变量解决问题(Type 0)
    2. 根据原始数据即可二值化的，创造2维哑变量解决问题 暂无
    3. 只需要定义为正常/异常的feature，也就是设定一个阈值，将数值型数据进行相应的转换 (Type 2)
    4. 需要定义为偏高/正常/偏低的feature，设定哑变量，将数值型数据创造为3维one hot编码 暂无

    若以后需要进行额外的细化，则进行类型拓展
    :param data_path:
    :return:
    """
    binary_dict = dict()
    with open(data_path, 'r', encoding='gbk', newline='') as file:
        csv_reader = csv.reader(file)
        for line in csv_reader:
            if line[1] == '0':
                binary_dict[line[0]] = [0, ]
            elif line[1] == "1":
                binary_dict[line[0]] = [1, float(line[2]), float(line[3])]
            elif line[1] == '2':
                binary_dict[line[0]] = [2, float(line[2])]
    return binary_dict


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


def binary(data_dict, binary_process_dict, contextual_list):
    """
    根据配置文件进行二值化
    :return:
    """
    item_list = list()

    # 读取item，变换数据结构
    for item in binary_process_dict:
        item_list.append(item)
        contextual_list.remove(item)
        if binary_process_dict[item][0] == 0:
            for patient_id in data_dict:
                for visit_id in data_dict[patient_id]:
                    data_dict[patient_id][visit_id][item + "_1"] = 0
                    data_dict[patient_id][visit_id][item + "_2"] = 0
                    data_dict[patient_id][visit_id][item + "_3"] = 0
                    data_dict[patient_id][visit_id][item + "_lost"] = 0

                    value = float(data_dict[patient_id][visit_id].pop(item))
                    if value == -1 or value == -1.0:
                        data_dict[patient_id][visit_id][item + "_lost"] = 1
                    elif value == 0:
                        data_dict[patient_id][visit_id][item + "_1"] = 1
                    elif value == 1:
                        data_dict[patient_id][visit_id][item + "_2"] = 1
                    else:
                        data_dict[patient_id][visit_id][item + "_3"] = 1
            contextual_list.append(item + "_1")
            contextual_list.append(item + "_2")
            contextual_list.append(item + "_3")
            contextual_list.append(item + "_lost")
        if binary_process_dict[item][0] == 1:
            for patient_id in data_dict:
                for visit_id in data_dict[patient_id]:
                    data_dict[patient_id][visit_id][item + "_1"] = 0
                    data_dict[patient_id][visit_id][item + "_2"] = 0
                    data_dict[patient_id][visit_id][item + "_3"] = 0
                    data_dict[patient_id][visit_id][item + "_lost"] = 0

                    value = float(data_dict[patient_id][visit_id].pop(item))
                    if value == -1 or value == -1.0:
                        data_dict[patient_id][visit_id][item + "_lost"] = 1
                    elif value < float(binary_process_dict[item][1]):
                        data_dict[patient_id][visit_id][item + "_1"] = 1
                    elif value < float(binary_process_dict[item][2]):
                        data_dict[patient_id][visit_id][item + "_2"] = 1
                    else:
                        data_dict[patient_id][visit_id][item + "_3"] = 1
            contextual_list.append(item + "_1")
            contextual_list.append(item + "_2")
            contextual_list.append(item + "_3")
            contextual_list.append(item + "_lost")
        if binary_process_dict[item][0] == 2:
            for patient_id in data_dict:
                for visit_id in data_dict[patient_id]:
                    data_dict[patient_id][visit_id][item + "_1"] = 0
                    data_dict[patient_id][visit_id][item + "_2"] = 0
                    data_dict[patient_id][visit_id][item + "_lost"] = 1
                    value = float(data_dict[patient_id][visit_id].pop(item))
                    if value == -1 or value == -1.0:
                        data_dict[patient_id][visit_id][item + "_lost"] = 1
                    elif value < binary_process_dict[item][1]:
                        data_dict[patient_id][visit_id][item + "_1"] = 1
                    elif value < binary_process_dict[item][2]:
                        data_dict[patient_id][visit_id][item + "_2"] = 1
            contextual_list.append(item + "_1")
            contextual_list.append(item + "_2")
            contextual_list.append(item + "_lost")
    return data_dict, contextual_list


def read_data(file_path, event_count=11):
    data_dict = dict()
    feature_dict = dict()
    event_index_list = list()
    context_index_list = list()
    with open(file_path, 'r', encoding='gbk', newline='') as file:
        csv_reader = csv.reader(file)
        head_flag = True
        for line in csv_reader:
            if head_flag:
                for i in range(2, len(line)):
                    feature_dict[i] = line[i]
                    # 按照需求记录event, context的有序内容信息
                    if i <= event_count+3:
                        event_index_list.append(line[i])
                    else:
                        context_index_list.append(line[i])
                head_flag = False
                continue

            patient_id = line[0]
            visit_id = line[1]

            if not data_dict.__contains__(patient_id):
                data_dict[patient_id] = dict()
            data_dict[patient_id][visit_id] = dict()

            for i in range(2, len(line)):
                data_dict[patient_id][visit_id][feature_dict[i]] = line[i]
    return data_dict, event_index_list, context_index_list


if __name__ == '__main__':
    main()
