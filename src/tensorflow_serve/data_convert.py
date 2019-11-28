import os
import json
import datetime
import re
import csv
import numpy as np
import sys

lab_test_range_dict = {
    '血清游离T4测定': {'low_threshold': 10.42, 'up_threshold': 24.32, 'unit': 'g/L'},
    '血清三碘甲腺原氨酸测定': {'low_threshold': 1.01, 'up_threshold': 2.95, "unit": "nmol/L"},
    '载脂蛋白A1': {'low_threshold': 1.0, 'up_threshold': 1.6, 'unit': 'g/L'},
    '血清尿酸': {'low_threshold': 104, 'up_threshold': 444, "unit": "umol/L"},
    '国际标准化比值': {'low_threshold': 0.95, 'up_threshold': 1.5, "unit": "noUnit", "comment": "此处301医院的原始数据较为杂乱，划分标准可能存在偏差"},
    '总胆红素': {'low_threshold': 0, 'up_threshold': 21.0, "unit": "umol/L"},
    '无机磷': {'low_threshold': 0.89, 'up_threshold': 1.6, "unit": "mmol/L"},
    '红细胞计数': {'low_threshold': 4.3, 'up_threshold': 5.9, "unit": "10^12/L"},
    '钾': {'low_threshold': 3.5, 'up_threshold': 5.5, "unit": "mmol/L"},
    '乳酸脱氢酶': {'low_threshold': 40, 'up_threshold': 250, "unit": "U/L"},
    '肌酸激酶同工酶定量测定': {'low_threshold': 0, 'up_threshold': 6.5, "unit": "ng/ml"},
    '总胆固醇': {'low_threshold': 3.1, 'up_threshold': 5.7, "unit": "mmol/L"},
    '高密度脂蛋白胆固醇': {'low_threshold': 1.0, 'up_threshold': 1.6, "unit": "mmol/L"},
    '血浆纤维蛋白原测定': {'low_threshold': 2.0, 'up_threshold': 4.0, "unit": "g/L"},
    '嗜酸性粒细胞': {'low_threshold': 0.01, 'up_threshold': 0.05, "unit": "%"},
    '血浆活化部分凝血酶原时间测定': {'low_threshold': 30, 'up_threshold': 45, "unit": "s"},
    '肌酐': {'low_threshold': 30, 'up_threshold': 110, "unit": "umol/L"},
    '葡萄糖': {'low_threshold': 3.4, 'up_threshold': 6.1, "unit": "mmol/L"},
    '天冬氨酸氨基转移酶': {'low_threshold': 0, 'up_threshold': 40, "unit": "U/L"},
    '单核细胞': {'low_threshold': 0.03, 'up_threshold': 0.08, "unit": "%"},
    '尿素': {'low_threshold': 1.8, 'up_threshold': 7.5, "unit": "mmol/L"},
    '直接胆红素': {'low_threshold': 0, 'up_threshold': 8.6, "unit": "umol/L"},
    '淋巴细胞': {'low_threshold': 0.2, 'up_threshold': 0.4, "unit": "%"},
    '血浆凝血酶原时间测定': {'low_threshold': 11, 'up_threshold': 15, "unit": "s"},
    '载脂蛋白B': {'low_threshold': 0.6, 'up_threshold': 1.1, "unit": "g/L"},
    '白细胞计数': {'low_threshold': 3.5, 'up_threshold': 10, "unit": "10^9/L"},
    '血小板计数': {'low_threshold': 100, 'up_threshold': 300, "unit": "10^9g/L"},
    '血红蛋白测定': {'low_threshold': 137, 'up_threshold': 179,"unit": "g/L"},
    '血清胱抑素(Cystatin C)测定': {'low_threshold': 0.45, 'up_threshold': 1.25, "unit": "mg/L"},
    '钠': {'low_threshold': 130, 'up_threshold': 150, "unit": "mmol/L"},
    '红细胞比积测定': {'low_threshold': 0.4, 'up_threshold': 0.52, "unit": "%"},
    '血浆D-二聚体测定': {'low_threshold': 0, 'up_threshold': 0.5, "unit": "ug/ml"},
    '血清游离T3测定': {'low_threshold': 2.76, 'up_threshold': 6.3, "unit": "pmol/L"},
    '平均血小板体积测定': {'low_threshold': 6.8, 'up_threshold': 12.8, "unit": "fl"},
    '血清甲状腺素测定': {'low_threshold': 55.34, 'up_threshold': 160.88, "unit": "nmol/L"},
    '钙': {'low_threshold': 2.09, 'up_threshold': 2.54, "unit": "mmol/L"},
    '脑利钠肽前体': {'low_threshold': 0, 'up_threshold': 150, "unit": "pg/ml"},
    'C-反应蛋白测定': {'low_threshold': 0, 'up_threshold': 0.8, "unit": "mg/dl"},
    '中性粒细胞': {'low_threshold': 0.5, 'up_threshold': 0.7, "unit": "%"},
    '丙氨酸氨基转移酶': {'low_threshold': 0, 'up_threshold': 40, "unit": "U/L"},
    '肌酸激酶': {'low_threshold': 2, 'up_threshold': 200, "unit": "U/L"},
    'γ-谷氨酰基转移酶': {'low_threshold': 0, 'up_threshold': 50, "unit": "U/L"},
    '总蛋白': {'low_threshold': 55, 'up_threshold': 80, "unit": "g/L"},
    '肌钙蛋白T': {'low_threshold': 0, 'up_threshold': 0.1, "unit": "ng/ml"},
    '血浆凝血酶原活动度测定': {'low_threshold': 70, 'up_threshold': 150, "unit": "%"},
    '碱性磷酸酶': {'low_threshold': 0, 'up_threshold': 130, "unit": "U/L"},
    '甘油三酯': {'low_threshold': 0.4, 'up_threshold': 1.7, "unit": "mmol/L"},
    '嗜碱性粒细胞': {'low_threshold': 0, 'up_threshold': 0.01, "unit": "%"},
    '低密度脂蛋白胆固醇': {'low_threshold': 0, 'up_threshold': 3.4, "unit": "null"},
    '镁': {'low_threshold': 0.6, 'up_threshold': 1.4, "unit": "mmol/L"},
    '血清促甲状腺激素测定': {'low_threshold': 0.35, 'up_threshold': 5.50, "unit": "mU/L"},
    '血清白蛋白': {'low_threshold': 35, 'up_threshold': 50, "unit": "g/L"},
    '凝血酶时间测定': {'low_threshold': 15, 'up_threshold': 21, "unit": "s"},
    '红细胞体积分布宽度测定CV': {'low_threshold': 0, 'up_threshold': 14.5, "unit": "%"},
}

lab_test_list = [
    '脑利钠肽前体', '丙氨酸氨基转移酶', '天冬氨酸氨基转移酶', '总蛋白', '血清白蛋白', '总胆红素', '直接胆红素',
    '葡萄糖', '尿素', '肌酐', '血清尿酸', '钠', '钙', '无机磷', '镁', '钾', 'γ-谷氨酰基转移酶', '碱性磷酸酶',
    '肌酸激酶', '肌钙蛋白T', '乳酸脱氢酶', '总胆固醇', '甘油三酯', '高密度脂蛋白胆固醇', '低密度脂蛋白胆固醇',
    '红细胞计数', '血小板计数', '血红蛋白测定', '红细胞比积测定', '红细胞体积分布宽度测定CV', '白细胞计数',
    '中性粒细胞', '淋巴细胞', '单核细胞', '嗜酸性粒细胞', '嗜碱性粒细胞', '平均血小板体积测定', 'C-反应蛋白测定',
    '凝血酶时间测定', '血浆凝血酶原时间测定', '血浆凝血酶原活动度测定', '血浆纤维蛋白原测定', '血浆D-二聚体测定']

medicine_list = ["ARB", "抗凝药物", "ACEI", "beta-blocker", "强心药", "抗血小板", "钙通道阻滞剂", "利尿剂",
                 "扩血管药", "抗酸剂", "他汀"]

operation_list = ["主动脉球囊反搏术", "瓣膜置换", "植入心脏起搏器", "冠状动脉旁路移植术", "植入心脏除颤器",
                  "心脏射频消融术", "PCI", "冠脉造影"]

diagnosis_list = ["高脂血症", "心包疾病", "脑梗/脑出血", "糖尿病", "心率失常", "贫血", "心力衰竭", "瓣膜病",
                  "肾功能不全", "甲亢或甲减", "心肌梗死", "先天性心脏病", "高血压", "心肌病", "冠心病"]
task_mapping = {
    "oneYearRenalDisease": 1,
    "oneYearOther": 2,
    "oneYearNYHAClass1": 3,
    "oneYearNYHAClass2": 4,
    "oneYearNYHAClass3": 5,
    "oneYearNYHAClass4": 6,
    "oneYearDeath": 7,
    "oneYearRevascular": 8,
    "oneYearCancer": 9,
    "oneYearLungDisease": 10,
    "threeMonthRenalDisease": 1,
    "threeMonthOther": 2,
    "threeMonthNYHAClass1": 3,
    "threeMonthNYHAClass2": 4,
    "threeMonthNYHAClass3": 5,
    "threeMonthNYHAClass4": 6,
    "threeMonthDeath": 7,
    "threeMonthRevascular": 8,
    "threeMonthCancer": 9,
    "threeMonthLungDisease": 10
}

def json_parse(json_str, target_sequence_length=10):
    content = json.loads(json_str)
    key_list = list(content.keys())
    key_list.sort()
    first_visit_time_str = content[key_list[0]]['visitInfo']['admissionTime']
    first_visit_time = datetime.datetime.strptime(first_visit_time_str, '%Y-%m-%d %H:%M:%S.%f')

    feature_sequence = list()
    time_interval_list = list()
    event_list = list()
    for i, visit in enumerate(key_list):
        visit_all_info = content[visit]
        lab_test_info = parse_labtests(visit_all_info['labTests'])
        medicines = parse_medicines(visit_all_info['medicines'],
                                    discharge_time=visit_all_info['visitInfo']['dischargeTime'])
        operations = parse_operations(visit_all_info['operations'])
        diagnoses = parse_diagnoses(visit_all_info['diagnoses'])
        vital_signs = parse_vital_signs(visit_all_info['vitalSigns'])
        exams = parse_exams(visit_all_info['exams'])

        general_info = parse_general_info(visit_all_info['basicInfo'], visit_all_info['visitInfo'],
                                          first_visit_time)
        current_event, cardiovascular_flag = \
            parse_events(visit_all_info['diagnoses'], visit_all_info['operations'])
        egfr = calculate_egfr(visit_all_info['labTests'], visit_all_info['basicInfo'],
                              visit_all_info['visitInfo'])
        composed_vector = compose_to_one_vector(lab_test_info, medicines, operations, diagnoses,
                                                vital_signs, exams, general_info, cardiovascular_flag,
                                                egfr)
        feature_sequence.append(composed_vector)
        time_interval_list.append(general_info['time_interval'])

        event_list.append(current_event)

    sequence_length = len(key_list)

    feature_sequence = np.array(feature_sequence)
    time_interval_list = np.array(time_interval_list)
    event_list = np.array(event_list)
    if len(feature_sequence) > target_sequence_length:
        feature_sequence = feature_sequence[(len(feature_sequence)-target_sequence_length): len(feature_sequence)]
        time_interval_list = time_interval_list[(len(time_interval_list) - target_sequence_length): len(time_interval_list)]
        event_list = event_list[(len(event_list) - target_sequence_length): len(event_list)]
    elif len(feature_sequence) < target_sequence_length:
        padding_length = target_sequence_length - len(feature_sequence)

        padding_feature = np.zeros([padding_length, len(feature_sequence[0])])
        padding_event = np.zeros([padding_length, len(event_list[0])])
        padding_time = np.zeros([padding_length])

        feature_sequence = np.concatenate([feature_sequence, padding_feature], axis=0)
        event_list = np.concatenate([event_list, padding_event], axis=0)
        time_interval_list = np.concatenate([time_interval_list, padding_time], axis=0)

    # 升维是为了适配batch size维度，在测试环境下该值为1
    feature_sequence = np.expand_dims(feature_sequence, axis=1).tolist()
    event_list = np.expand_dims(event_list, axis=1).tolist()
    time_interval_list = np.expand_dims(time_interval_list, axis=0).tolist()
    sequence_length = np.array(sequence_length).tolist()
    return sequence_length, feature_sequence, time_interval_list, event_list


def compose_to_one_vector(lab_test_info, medicines, operations, diagnoses, vital_signs,
                          exams, general_info, cardiovascular_flag, egfr):
    composed_vector = list()
    composed_vector.append(1 if cardiovascular_flag else 0)
    composed_vector.append(general_info['sex'])
    for item in operations:
        composed_vector.append(item)
    for item in medicines:
        composed_vector.append(item)
    for item in diagnoses:
        composed_vector.append(item)
    for item in lab_test_info:
        composed_vector.append(item)
    for item in vital_signs['dbp']:
        composed_vector.append(item)
    for item in exams:
        composed_vector.append(item)
    for item in vital_signs['sbp']:
        composed_vector.append(item)
    for item in vital_signs['hp']:
        composed_vector.append(item)
    for item in egfr:
        composed_vector.append(item)
    for item in general_info['age']:
        composed_vector.append(item)
    for item in general_info['los']:
        composed_vector.append(item)
    for item in vital_signs['bmi']:
        composed_vector.append(item)
    return composed_vector


def calculate_egfr(labtest, basic_info, visit_info):
    sex = 1 if basic_info['sex'] == "男" else 0
    current_visit_time_str = visit_info['admissionTime']
    current_visit_time = datetime.datetime.strptime(current_visit_time_str, '%Y-%m-%d %H:%M:%S.%f')
    birthday_str = basic_info['birthday']
    birthday = datetime.datetime.strptime(birthday_str, '%Y-%m-%d %H:%M:%S.%f')
    age = (current_visit_time-birthday).days / 365

    scr = -1
    for item in labtest:
        if item['labTestItemName'] == '肌酐':
            result = item['result']

            result_list = re.findall('[-+]?[\d]+(?:,\d\d\d)*[.]?\d*(?:[eE][-+]?\d+)?', result)
            if len(result_list) > 0:
                result = result_list[0]
            if len(result_list) == 0 or len(result) == 0:
                continue
            scr = float(result)

    if sex == 1.0:
        egfr = 186 * ((scr / 88.41) ** -1.154) * (age ** -0.203) * 1
    else:
        egfr = 186 * ((scr / 88.41) ** -1.154) * (age ** -0.203) * 0.742

    if scr == -1:
        return [0, 0, 0]

    if egfr > 90:
        egfr_vector = [0, 0, 1]
    elif egfr > 30:
        egfr_vector = [0, 1, 0]
    elif egfr > 0:
        egfr_vector = [1, 0, 0]
    else:
        egfr_vector = [0, 0, 0]
    return egfr_vector


def parse_events(diagnosis, operation):
    # 读取诊断，通过病人的出院诊断，判断是否心源性再入院
    cardiovascular_flag = False

    for item in diagnosis:
        if not item['key']['diagnosisType'] == "3":
            continue
        diagnosis_desc = item['diagnosisDesc']
        cardiovascular_flag = \
            diagnosis_desc.__contains__('心') or diagnosis_desc.__contains__('高血压') or \
            diagnosis_desc.__contains__('冠') or diagnosis_desc.__contains__('房') or \
            diagnosis_desc.__contains__('瓣') or diagnosis_desc.__contains__('室') or \
            diagnosis_desc.__contains__('膜') or diagnosis_desc.__contains__('先天性') or \
            diagnosis_desc.__contains__('房间隔') or diagnosis_desc.__contains__('室间隔') or \
            diagnosis_desc.__contains__('四联症') or diagnosis_desc.__contains__('肺动脉瓣狭窄') or \
            diagnosis_desc.__contains__('动脉导管未闭') or diagnosis_desc.__contains__('三尖瓣下移') or \
            diagnosis_desc.__contains__('主动脉缩窄') or diagnosis_desc.__contains__('二叶主动脉瓣') or \
            diagnosis_desc.__contains__('NYHA')

    # event
    # 读取诊断，判断死亡与心功能分级以及常见的合并症（糖尿病、肾病、肺病、癌症）
    diagnosis_dict = {'糖尿病': 0, '肾病': 0, '其它': 0, '癌症': 0, '肺病': 0,
                      '心功能1级': 0, '心功能2级': 0, '心功能3级': 0, '心功能4级': 0,
                      '死亡': 0, '心功能标签丢失': 0}

    for item in diagnosis:
        diagnosis_type = item['key']['diagnosisType']
        diagnosis_desc = item['diagnosisDesc']

        # 标定死亡
        if diagnosis_desc == '死亡':
            diagnosis_dict['死亡'] = 1

        # 标定心功能分级，由于心功能分级标定较为简单，不一定要通过出院诊断才能标记，因此不做主诊断要求
        if diagnosis_desc.__contains__('心功能'):
            pos = diagnosis_desc.find('心功能')
            target_length = len('心功能')
            if len(diagnosis_desc) - pos > 4:
                sub_string = diagnosis_desc[pos + target_length: pos + target_length + 4]
            else:
                sub_string = diagnosis_desc[pos + target_length:]
            if sub_string.__contains__('1') or sub_string.__contains__('I') or sub_string.__contains__('i') or \
                    sub_string.__contains__('一') or sub_string.__contains__('Ⅰ'):
                diagnosis_dict['心功能1级'] = 1
            elif sub_string.__contains__('2') or sub_string.__contains__('II') or sub_string.__contains__('ii') or \
                    sub_string.__contains__('二') or sub_string.__contains__('Ⅱ'):
                diagnosis_dict['心功能2级'] = 1
            elif sub_string.__contains__('3') or sub_string.__contains__('III') or \
                    sub_string.__contains__('iii') or sub_string.__contains__('三') or sub_string.__contains__('Ⅲ'):
                diagnosis_dict['心功能3级'] = 1
            elif sub_string.__contains__('4') or sub_string.__contains__('IV') or sub_string.__contains__('iv') or \
                    sub_string.__contains__('四') or sub_string.__contains__('Ⅳ'):
                diagnosis_dict['心功能4级'] = 1
            else:
                # 单纯写心功能不全的，统一按照二级处理
                diagnosis_dict['心功能2级'] = 1
        elif diagnosis_desc.__contains__('NYHA'):
            pos = diagnosis_desc.find('NYHA')
            target_length = len('NYHA')
            if len(diagnosis_desc) - pos > 4:
                sub_string = diagnosis_desc[pos + target_length: pos + target_length + 4]
            else:
                sub_string = diagnosis_desc[pos + target_length:]
            if sub_string.__contains__('1') or sub_string.__contains__('I') or sub_string.__contains__('i') or \
                    sub_string.__contains__('一') or sub_string.__contains__('Ⅰ'):
                diagnosis_dict['心功能1级'] = 1
            elif sub_string.__contains__('2') or sub_string.__contains__('II') or sub_string.__contains__('ii') or \
                    sub_string.__contains__('二') or sub_string.__contains__('Ⅱ'):
                diagnosis_dict['心功能2级'] = 1
            elif sub_string.__contains__('3') or sub_string.__contains__('III') or \
                    sub_string.__contains__('iii') or sub_string.__contains__('三') or sub_string.__contains__('Ⅲ'):
                diagnosis_dict['心功能3级'] = 1
            elif sub_string.__contains__('4') or sub_string.__contains__('IV') or sub_string.__contains__('iv') or \
                    sub_string.__contains__('四') or sub_string.__contains__('Ⅳ'):
                diagnosis_dict['心功能4级'] = 1
            else:
                # 单纯写心功能不全的，统一按照二级处理
                diagnosis_dict['心功能2级'] = 1

        # 根据出院诊断判断常见合并症是否发生
        if diagnosis_type == '3' and \
                (diagnosis_desc.__contains__('肾') or diagnosis_desc.__contains__('尿毒症')):
            diagnosis_dict['肾病'] = 1
        if diagnosis_type == '3' and \
                (diagnosis_desc.__contains__('瘤') or diagnosis_desc.__contains__('癌')):
            diagnosis_dict['癌症'] = 1
        if diagnosis_type == '3' and \
                (diagnosis_desc.__contains__('呼') or diagnosis_desc.__contains__('肺') or
                 diagnosis_desc.__contains__('气管') or diagnosis_desc.__contains__('吸')):
            diagnosis_dict['肺病'] = 1
        if diagnosis_type == '3' and (diagnosis_desc.__contains__('糖尿病')):
            diagnosis_dict['糖尿病'] = 1

    # 如果是心源性入院，但是诊断中没有标定心功能，则打上心功能标签丢失的Tag
    # 如果是非心源性入院，且不是上述四种疾病，则打上其它的Tag
    one = diagnosis_dict['心功能1级']
    two = diagnosis_dict['心功能2级']
    three = diagnosis_dict['心功能3级']
    four = diagnosis_dict['心功能4级']
    kidney = diagnosis_dict['肾病']
    diabetes = diagnosis_dict['糖尿病']
    cancer = diagnosis_dict['癌症']
    lung = diagnosis_dict['肺病']

    heart_class = one + two + three + four
    commodities = kidney + diabetes + cancer + lung
    # 判断是否出现了标签缺失（如果的确是心源性入院，但是没有心功能分级）
    if heart_class == 0 and cardiovascular_flag:
        diagnosis_dict['心功能标签丢失'] = 1
    elif (not cardiovascular_flag) and commodities == 0:
        diagnosis_dict['其它'] = 1

    # 读取手术，判断病人是否存在血运重建手术
    # 所谓血运重建手术，指PCI和CABG两种
    operation_flag = False
    candidate_name_list = ['支架', '球囊', 'PCI', '介入', 'PTCA', '冠状动脉旁路移植术', '旁路移植', 'CABG', '搭桥']
    for item in operation:
        for candidate_name in candidate_name_list:
            if item['operationDesc'].__contains__(candidate_name):
                operation_flag = True

    # 建立事件数据模板，保证一次入院只对应一个事件
    # 因此下面的列表代表了多个事件同时发生时的判定优先级
    event_dict = {'糖尿病入院': 0, '肾病入院': 0, '其它': 0, '心功能1级': 0, '心功能2级': 0, '心功能3级': 0,
                  '心功能4级': 0, '死亡': 0, '心功能标签丢失': 0, '再血管化手术': 0, '癌症': 0, '肺病': 0}
    single_diagnosis_dict = diagnosis_dict
    if single_diagnosis_dict['死亡'] == 1:
        event_dict['死亡'] = 1
    elif operation_flag:
        event_dict['再血管化手术'] = 1
    # 此处原本的设计是，如果是心源性入院的，才判定分级
    # 现在修改为，只要没有再血管化手术，死亡，有分级的，就判定为心功能入院
    # 这一设计可能导致有些病人并非心源性入院，但是有心功能分级的，被判定为心功能入院伴随分级
    # 在此，认为心功能分级有更高的判定优先级
    # 如果心功能标签丢失，认定为2级
    elif single_diagnosis_dict['心功能4级'] == 1:
        event_dict['心功能4级'] = 1
    elif single_diagnosis_dict['心功能3级'] == 1:
        event_dict['心功能3级'] = 1
    elif single_diagnosis_dict['心功能2级'] == 1:
        event_dict['心功能2级'] = 1
    elif single_diagnosis_dict['心功能1级'] == 1:
        event_dict['心功能1级'] = 1
    elif single_diagnosis_dict['心功能标签丢失'] == 1:
        event_dict['心功能2级'] = 1
    else:
        if single_diagnosis_dict['癌症'] == 1:
            event_dict['癌症'] = 1
        elif single_diagnosis_dict['肾病'] == 1:
            event_dict['肾病入院'] = 1
        elif single_diagnosis_dict['糖尿病'] == 1:
            event_dict['糖尿病入院'] = 1
        elif single_diagnosis_dict['肺病'] == 1:
            event_dict['肺病'] = 1
        elif single_diagnosis_dict['其它'] == 1:
            event_dict['其它'] = 1

    event_list = ["糖尿病入院", "肾病入院"	, "其它"	, "心功能1级", "心功能2级", "心功能3级",
                  "心功能4级", "死亡", "再血管化手术", "癌症", "肺病"]
    event_vector = list()
    for item in event_list:
        event_vector.append(event_dict[item])

    return event_vector, cardiovascular_flag


def parse_labtests(labtests_info):
    required_item_set = set(lab_test_list)

    # initialize
    lab_test_result = dict()
    for key in required_item_set:
        lab_test_result[key] = [-1, datetime.datetime(2020, 1, 1, 0, 0, 0, 0)]

    # read data
    for item in labtests_info:
        lab_test_item_name = item['labTestItemName']
        if not required_item_set.__contains__(lab_test_item_name):
            continue

        result = item['result']
        result_list = re.findall('[-+]?[\d]+(?:,\d\d\d)*[.]?\d*(?:[eE][-+]?\d+)?', result)
        if len(result_list) > 0:
            result = result_list[0]
        if len(result_list) == 0 or len(result) == 0:
            continue
        result = float(result)

        execute_date = datetime.datetime.strptime(item['executeDate'], '%Y-%m-%d %H:%M:%S.%f')

        if execute_date < lab_test_result[lab_test_item_name][1]:
            lab_test_result[lab_test_item_name] = [result, execute_date]

    # discrete
    lab_test_vector = list()
    for item in lab_test_list:
        result, _ = lab_test_result[item]
        if result == -1:
            lab_test_vector.append(0)
            lab_test_vector.append(0)
            lab_test_vector.append(0)
        else:
            low_bound = lab_test_range_dict[item]['low_threshold']
            up_bound = lab_test_range_dict[item]['up_threshold']
            if result <= low_bound:
                lab_test_vector.append(1)
                lab_test_vector.append(0)
                lab_test_vector.append(0)
            if result >= up_bound:
                lab_test_vector.append(0)
                lab_test_vector.append(0)
                lab_test_vector.append(1)
            if low_bound < result < up_bound:
                lab_test_vector.append(0)
                lab_test_vector.append(1)
                lab_test_vector.append(0)
    return lab_test_vector


def parse_general_info(basic_info, visit_info, first_visit_time):
    sex = 1 if basic_info['sex'] == "男" else 0

    current_visit_time_str = visit_info['admissionTime']
    current_visit_time = datetime.datetime.strptime(current_visit_time_str, '%Y-%m-%d %H:%M:%S.%f')
    time_difference = (current_visit_time-first_visit_time).days

    birthday_str = basic_info['birthday']
    birthday = datetime.datetime.strptime(birthday_str, '%Y-%m-%d %H:%M:%S.%f')

    age = (current_visit_time-birthday).days / 365
    if age > 80:
        age_vector = [0, 0, 1]
    elif age > 60:
        age_vector = [0, 1, 0]
    elif age > 40:
        age_vector = [1, 0, 0]
    else:
        age_vector = [0, 0, 0]

    current_discharge_time_str = visit_info['dischargeTime']
    current_discharge_time = datetime.datetime.strptime(current_discharge_time_str, '%Y-%m-%d %H:%M:%S.%f')
    los = (current_discharge_time-current_visit_time).days
    if los > 15:
        los_vector = [0, 0, 1]
    elif los > 8:
        los_vector = [0, 1, 0]
    elif los > 0:
        los_vector = [1, 0, 0]
    else:
        los_vector = [0, 0, 0]
    return {'los': los_vector, 'age': age_vector, 'time_interval': time_difference, "sex": sex}


def parse_medicines(medicines_info, discharge_time):
    discharge_time = datetime.datetime.strptime(discharge_time, '%Y-%m-%d %H:%M:%S.%f')

    # 建立10类药物名称映射
    drug_map_path = os.path.abspath('../../resource/药品名称映射.csv')
    drug_map_dict = dict()
    drug_map_set = set()
    with open(drug_map_path, 'r', encoding='gbk', newline="") as file:
        csv_reader = csv.reader(file)
        for line in csv_reader:
            drug_map_set.add(line[0])
            for i in range(len(line)):
                if len(line[i]) >= 2:
                    drug_map_dict[line[i]] = line[0]
    # 建立模板
    drug_usage_map = dict()
    for item in drug_map_set:
        drug_usage_map[item] = datetime.datetime(1970, 1, 1, 0, 0, 0, 0)

    for item in medicines_info:
        stop_time = datetime.datetime.strptime(item['stopDateTime'], '%Y-%m-%d %H:%M:%S.%f')
        order_text = item['orderText']
        for key in drug_map_dict:
            if order_text.__contains__(key):
                if stop_time > drug_usage_map[drug_map_dict[key]]:
                    drug_usage_map[key] = stop_time

    medicine_vector = []
    for item in medicine_list:
        last_time = drug_usage_map[item]

        time_interval = discharge_time - last_time
        time_interval = (time_interval.days * 3600 * 24 + time_interval.seconds) / 3600
        if time_interval < 36:
            medicine_vector.append(1)
        else:
            medicine_vector.append(0)

    return medicine_vector


def parse_operations(operation_info):
    operation_map_path = os.path.abspath('../../resource/手术名称映射.csv')

    # 从外源性文件导入名称映射策略
    operation_name_set = set()
    operation_name_list = list()
    with open(operation_map_path, 'r', encoding='gbk', newline='') as file:
        csv_reader = csv.reader(file)
        for line in csv_reader:
            operation_name_set.add(line[0])
            row = []
            for i in range(0, len(line)):
                if len(line[i]) >= 1:
                    row.append(line[i])
            operation_name_list.append(row)

    # 建立数据模板
    operation_dict = dict()
    for item in operation_name_set:
        operation_dict[item] = 0

    # 填补
    for item in operation_info:
        operation_desc = item['operationDesc']
        for item_list in operation_name_list:
            for i in range(0, len(item_list)):
                if operation_desc.__contains__(item_list[i]):
                    operation_dict[item_list[0]] = 1

    operation_vector = list()
    for item in operation_list:
        operation_vector.append(operation_dict[item])
    return operation_vector


def parse_exams(exam_info):
    lvef = -1
    for item in exam_info:
        exam_para = item['examPara']
        if exam_para.__contains__("射血分数"):
            pos = exam_para.find("射血分数")
            target_length = len("射血分数")
            sub_string = exam_para[pos + target_length: pos + target_length + 5]
            value_list = re.findall('[-+]?[.]?[\d]+(?:,\d\d\d)*[.]?\d*(?:[eE][-+]?\d+)?', sub_string)
            if len(value_list) > 0:
                lvef = float(value_list[0])

    if lvef < 50:
        lvef_vector = [1, 0, 0]
    elif lvef >= 50:
        lvef_vector = [0, 1, 0]
    else:
        lvef_vector = [0, 0, 0]
    return lvef_vector


def parse_vital_signs(vital_signs_info):
    vital_sign_dict = {'血压Low': [-1, datetime.datetime(2020, 1, 1, 0, 0, 0, 0)],
                       '血压high': [-1, datetime.datetime(2020, 1, 1, 0, 0, 0, 0)],
                       '脉搏': [-1, datetime.datetime(2020, 1, 1, 0, 0, 0, 0)],
                       "身高": [-1, datetime.datetime(2020, 1, 1, 0, 0, 0, 0)],
                       "体重": [-1, datetime.datetime(2020, 1, 1, 0, 0, 0, 0)]}

    for item in vital_signs_info:
        record_time = datetime.datetime.strptime(item['recordTime'], '%Y-%m-%d %H:%M:%S.%f')
        vital_sign = item['key']['vitalSign']
        result = item['result']

        if not vital_sign_dict.__contains__(vital_sign):
            continue

        if record_time < vital_sign_dict[vital_sign][1]:
            vital_sign_dict[vital_sign] = [result, record_time]

    systolic_blood_pressure = vital_sign_dict['血压high'][0]
    if systolic_blood_pressure > 100:
        sbp_vector = [0, 0, 1]
    elif systolic_blood_pressure > 80:
        sbp_vector = [0, 1, 0]
    elif systolic_blood_pressure > 40:
        sbp_vector = [1, 0, 0]
    else:
        sbp_vector = [0, 0, 0]

    diastolic_blood_pressure = vital_sign_dict['血压Low'][0]
    if diastolic_blood_pressure > 100:
        dbp_vector = [0, 0, 1]
    elif diastolic_blood_pressure > 80:
        dbp_vector = [0, 1, 0]
    elif diastolic_blood_pressure > 40:
        dbp_vector = [1, 0, 0]
    else:
        dbp_vector = [0, 0, 0]

    heart_beat = vital_sign_dict['脉搏'][0]
    if heart_beat > 100:
        hb_vector = [0, 0, 1]
    elif heart_beat > 60:
        hb_vector = [0, 1, 0]
    elif heart_beat > 20:
        hb_vector = [1, 0, 0]
    else:
        hb_vector = [0, 0, 0]

    bmi = vital_sign_dict['体重'][0] / (vital_sign_dict['身高'][0])**2
    if bmi > 28:
        bmi_vector = [0, 0, 1]
    elif bmi > 24:
        bmi_vector = [0, 1, 0]
    elif bmi > 18:
        bmi_vector = [1, 0, 0]
    else:
        bmi_vector = [0, 0, 0]
    return {'sbp': sbp_vector, "dbp": dbp_vector, "bmi": bmi_vector, "hp": hb_vector}


def parse_diagnoses(diagnoses_info):
    diagnosis_map_path = os.path.abspath('../../resource/合并症不同名归一化.csv')

    # 通过外源性归一化清单，建立模板
    normalized_dict = dict()
    normalized_set = set()
    with open(diagnosis_map_path, 'r', encoding='gbk', newline="") as file:
        csv_reader = csv.reader(file)
        for line in csv_reader:
            normalized_set.add(line[0])
            for item in line:
                if len(item) >= 1:
                    normalized_dict[item] = line[0]
    diagnosis_dict = dict()
    for item in normalized_set:
        diagnosis_dict[item] = 0

    for single_diagnosis in diagnoses_info:
        diagnosis_desc = single_diagnosis['diagnosisDesc']

        # 进行数据填充
        for item in normalized_dict:
            if diagnosis_desc.__contains__(item):
                diagnosis_dict[normalized_dict[item]] = 1

    diagnosis_vector = list()
    for item in diagnosis_list:
        diagnosis_vector.append(diagnosis_dict[item])
    return diagnosis_vector


def time_parse(time_str):
    if len(time_str) > 12:
        time = datetime.datetime.strptime(time_str, '%Y/%m/%d %H:%M:%S')
    elif 4 < len(time_str) < 12:
        time = datetime.datetime.strptime(time_str, '%Y/%m/%d')
    else:
        time = datetime.datetime(1970, 1, 1, 0, 0, 0, 0)
        print('time illegal')
    return time


if __name__ == "__main__":
    json_str, task = sys.argv

    # json_example_file = os.path.abspath("../../resource/tensorflow_serve/json_template.txt")
    # with open(json_example_file, 'r', encoding='utf-8') as file:
    #    json_str = file.readline()
    # task = "threeMonthDeath"

    sequence_length, feature_sequence, time_interval_list, event_list = json_parse(json_str)

    mutual_intensity_path = os.path.abspath('../../resource/hawkes_result/mutual.npy')
    base_intensity_path = os.path.abspath('../../resource/hawkes_result/base.npy')
    mutual_intensity_value = np.load(mutual_intensity_path).tolist()
    base_intensity_value = np.load(base_intensity_path).tolist()

    inputs = {"event": event_list,
              "context": feature_sequence,
              "base": base_intensity_value,
              "batch": 1,
              "mutual": mutual_intensity_value,
              "phase": 1,
              "sequence_length": sequence_length,
              "time_list": time_interval_list,
              "task": task_mapping[task]}
    tensorflow_feed_data = json.dumps({'inputs': inputs})
    print(tensorflow_feed_data)
