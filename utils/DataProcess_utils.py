# coding: utf-8

import numpy as np

from keras.utils import to_categorical

from collections import Counter

import bisect

# from position in state to specific state species
state_dict = {
    0: "count", 1: "cycle", 2: "day_in_cycle", 3: "day_gap", 4: "user_tag", 5: "mord_actual_total_amt",
    6: "login_days_7", 7: "gender", 8: "age_level", 9: "is_marriage", 10:	"consume_level",
    11: "has_baby", 12: "baby_gender", 13: "baby_age", 14: "is_car", 15: "car_level", 16: "car_brand_id",
    17: "cat1_prefer", 18: "cat1_brand_prefer", 19: "brand_score", 20: "home_lat",
    21: "home_lng", 22: "home_x", 23: "home_y", 24: "home_prov", 25: "home_city", 26: "home_district",
    27: "work_lat", 28: "work_lng", 29: "work_x", 30: "work_y", 31: "work_prov", 32: "work_city",
    33: "work_district", 34: "community_lat", 35: "community_lng", 36: "is_miaojie", 37: "is_shoutao",
    38: "occupation", 39: "os", 40: "consume_level_2", 41: "vip_level", 42: "uc_fc_tag",
    43: "workaround_lat", 44: "workaround_lng", 45: "discount_sensitive", 46: "prefer_car_price",
    47: "prefer_car_tag", 48: "lbs_occupation2", 49: "city_name", 50: "industry_type",
    51: "education", 52: "pred_has_house", 53: "property_hourse_level", 54: "income_level",
    55: "city_income_level", 56: "prov_code"
}

not_considered_features = ["count", "cycle", "car_brand_id", "cat1_prefer", "cat1_brand_prefer",
                           "home_x", "home_y", "home_city", "home_district",
                           "work_x", "work_y", "work_city", "work_district",
                           "is_miaojie", "is_shoutao", "os", "city_name",
                           "prov_code", "prefer_car_price",
                           #
                           "gender", "age_level", "is_marriage", "consume_level",
                           "has_baby", "baby_gender", "baby_age", "is_car", "car_level", "car_brand_id",
                           "cat1_prefer", "cat1_brand_prefer", "brand_score", "home_lat",
                           "home_lng", "home_x", "home_y", "home_prov", "home_city", "home_district",
                           "work_lat", "work_lng", "work_x", "work_y", "work_prov", "work_city",
                           "work_district", "community_lat", "community_lng", "is_miaojie", "is_shoutao",
                           "occupation", "os", "consume_level_2", "vip_level", "uc_fc_tag",
                           "workaround_lat", "workaround_lng", "discount_sensitive", "prefer_car_price",
                           "prefer_car_tag", "lbs_occupation2", "city_name", "industry_type",
                           "education", "pred_has_house", "property_hourse_level", "income_level",
                           "city_income_level", "prov_code"
                           ]

user_tag_list = ["old_J0B1", "old_J0B2", "old_J1B1", "old_J1B2", "old_J2B1", "old_J2B2", "old_J3.1B1",
                 "old_J3.1B2", "old_J3B1", "old_J3B2", "old_J4B1", "old_J4B2", "old_J5B1", "old_J5B2",
                 "old_LS15B1", "old_LS15B2", "old_LS7B1", "old_LS7B2", "old_newLogin", "other"]

gender_list = ['0', '1', '2']

is_marriage_list = ['0', '1', '2']

has_baby_list = ['0', '1', '2']

baby_gender_list = ['0', '1', '2', '3', '4']

is_car_list = ['0', '1']

work_prov_list = home_prov_list = ["吉林省", "宁夏回族自治区", "山东省", "山西省", "湖北省", "辽宁省", "安徽省", "广西壮族自治区",
                                   "江西省", "河北省", "贵州省", "重庆市", "云南省", "天津市", "江苏省", "河南省", "浙江省",
                                   "湖南省", "西藏自治区", "黑龙江省", "上海市", "北京市", "台湾省", "广东省", "新疆维吾尔自治区",
                                   "澳门特别行政区", "福建省", "青海省", "内蒙古自治区", "四川省", "海南省", "甘肃省", "陕西省",
                                   "香港特别行政区", "未知"]

occupation_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '0']

uc_fc_tag_list = ['1', '0']

prefer_car_tag_list = ['1', '0']

lbs_occupation2_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-98']

industry_type_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
                      '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
                      '21', '22', '23', '24', '25', '26', '27', '28', '29', '30',
                      '31', '32', '33', '34', '35', '36', '-98']

education_list = ['0', '1', '2', '3', '4', '-98']

pred_has_house_list = ['0', '1', '2']

# from specific state to this state_list
state_specific_to_list = {"user_tag": user_tag_list, "gender": gender_list,
                          "is_marriage": is_marriage_list, "has_baby": has_baby_list,
                          "baby_gender": baby_gender_list, "is_car": is_car_list,
                          "home_prov": home_prov_list, "work_prov": work_prov_list,
                          "occupation": occupation_list, "uc_fc_tag": uc_fc_tag_list,
                          "prefer_car_tag": prefer_car_tag_list, "lbs_occupation2": lbs_occupation2_list,
                          "industry_type": industry_type_list, "education": education_list,
                          "pred_has_house": pred_has_house_list
                          }


action_list = list(np.array(
    [0.01, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 0.43,
     0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6,
     0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69, 0.7, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77,
     0.78, 0.79, 0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94,
     0.95, 0.96, 0.97, 0.98, 0.99, 1, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09, 1.1, 1.11, 1.12,
     1.13, 1.14, 1.15, 1.16, 1.17, 1.18, 1.19, 1.2, 1.21, 1.22, 1.23, 1.24, 1.25, 1.26, 1.27, 1.28, 1.29, 1.3,
     1.31, 1.32, 1.33, 1.34, 1.35, 1.36, 1.37, 1.38, 1.39, 1.4, 1.41, 1.42, 1.43, 1.44, 1.45, 1.46, 1.47,
     1.48, 1.49, 1.5, 1.51, 1.52, 1.53, 1.54, 1.55, 1.56, 1.57, 1.58, 1.59, 1.6, 1.61, 1.62, 1.63, 1.64,
     1.65, 1.66, 1.67, 1.68, 1.69, 2], dtype=np.float32))

action_mapping_liucun_rate_mapping = {
    (0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.6, 0.61,
     0.62, 0.63, 0.64): -0.087812989,
    0.65: -0.242240099,
    (0.66, 0.67, 0.68, 0.69, 0.7, 0.71, 0.72, 0.73, 0.74): -0.126906367,
    0.75: -0.208548725,
    (0.76, 0.77, 0.78, 0.79, 0.8, 0.81, 0.82, 0.83, 0.84, 0.85): -0.158576658,
    0.86: -0.10181997,
    (0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98,
     0.99): -0.321138779,
    1.0: -0.49544049,
    (1.01, 1.02, 1.03, 1.04, 1.05): -0.312547656,
    (1.09, 1.1, 1.11): -0.189647313,
    (1.14, 1.15, 1.16, 1.17): -0.081714423,
    (1.18, 1.19, 1.2): -0.133112561,
    (1.21, 1.22, 1.23, 1.24): -0.193048087,
    1.25: -0.259410053,
    (1.26, 1.27): -0.138436025,
    1.28: -0.07907,
    1.35: -0.209795192
}
# action_mapping_action_num_mapping = {
#     (0.5, 0.51, 0.56): 0,
#     (0.65, 0.7, 0.71, ): 1,
#     (0.75, ): 2,
#     (0.77, 0.79, 0.81, 0.84, ): 3,
#     (0.87, 1, 1.05, ): 4,
#     (1.16, ): 5,
#     (1.21, 1.25, 1.27, ): 6,
#     (0.52, 0.53, 0.54, 0.55, 0.58, 0.59, 0.61, ): 7,
#     (0.65, ): 8,
#     (0.67, 0.68, 0.69, 0.72, 0.73, 0.74, ): 9,
#     (0.76, 0.78, ): 10,
#     (0.8, 0.82, 0.83, 0.85, ): 11,
#     (0.86, ): 12,
#     (0.88, ): 13,
#     (1.14, 1.15, 1.17, ): 14,
#     (1.19, ): 15,
#     (1.22, 1.23, 1.24, ): 16,
#     (1.26, ): 17,
#     (1.28, ): 18,
#     (0.57, 0.6, 0.62, 0.63, 0.64, ): 19,
# }
action_mapping_action_num_mapping = {
    # (0.5, ): 0,
    # (0.65, ): 1,
    # (0.66, ): 2,
    # (0.68, ): 3,
    # (0.69, ): 4,
    # (0.7, ): 5,
    # (0.75, ): 6,
    # (0.86, ): 7,
    # (1.09,): 8,
    # (1.11, ): 9,
    (0.1, ): 0,
    (0.2, ): 1,
    (0.3, ): 2,
    (0.4, ): 3,
    (0.5, ): 4,
    (0.6, ): 5,
    (0.7, ): 6,
    (0.8, ): 7,
    (0.9, ): 8,
    (1.0, ): 9,
    (1.1, ): 10,
    (1.2, ): 11,
    (1.3, ): 12,
    (1.4, ): 13,
    (1.5, ): 14,
    (1.6, ): 15,
    (1.7, ): 16,
    (1.8, ): 17,
    (1.9, ): 18,
    (2.0, ): 19,
    (2.1, ): 20,
}
#
# action_mapping_action_num_mapping = {
#     (np.float32(0.00), ): 0,
#     (np.float32(0.10), ): 1,
#     (np.float32(0.20), ): 2,
#     (np.float32(0.30), ): 3,
#     (np.float32(0.40), ): 4,
# }

regularize_list = {"home_lat": 0, "home_lng": 1, "work_lat": 2, "work_lng": 3, "community_lat": 4, "community_lng": 5,
                   "workaround_lat": 6, "workaround_lng": 7}
regularization = [91./10., 181./10., 91./10., 181./10., 91./10., 181./10., 91./10., 181./10.]

action_mapping_liucun_rate_dict = {}
for k, v in action_mapping_action_num_mapping.items():
    for key in k:
        action_mapping_liucun_rate_dict[np.float32(key)] = v

sorted_keys = sorted(list(action_mapping_liucun_rate_dict.keys()))
len_sorted_keys = len(sorted_keys)
for action in action_list:
    if action not in np.float32(sorted_keys):
        i = min(bisect.bisect_left(sorted_keys, action), len_sorted_keys - 1)
        action_mapping_liucun_rate_dict[action] = action_mapping_liucun_rate_dict[sorted_keys[i]]
action_mapping_back_liucun_rate_dict = {}
for k, num in action_mapping_action_num_mapping.items():
    action_mapping_back_liucun_rate_dict[num] = np.mean(list(k))


def embedding_state(state, data):
    if state in not_considered_features:
        return None
    if state in state_specific_to_list:
        state_list = state_specific_to_list[state]
        mapping = {}
        for x in range(len(state_list)):
            mapping[state_list[x]] = x

        # integer representation
        res_one_hot_data = np.zeros_like(data)
        for x in range(len(data)):
            if data[x] not in mapping:
                data[x] = mapping[state_list[-1]]
            res_one_hot_data[x] = mapping[data[x]]

        return to_categorical(res_one_hot_data, num_classes=len(state_list))
    else:
        # We don't need to implement this kind of state
        if state in regularize_list:
            return np.expand_dims(data.astype(np.float32) / regularization[regularize_list[state]], axis=1)
        else:
            return np.expand_dims(data, axis=1)


class ActionEmbedding:
    def __init__(self, action_dim=8):
        self.is_called = False
        self.mapping = {}
        self.mapping_back = {}
        self.action_dim = action_dim

    def action_embedding(self, data):
        if self.is_called:
            return self._action(data)
        else:
            self.is_called = True
            return self._action_embedding(data)

    def _action(self, data):
        if self.mapping == {}:
            raise NotImplementedError
        mapping = self.mapping

        # integer representation
        res_one_hot_data = np.zeros_like(data)
        for x in range(len(data)):
            res_one_hot_data[x] = mapping[data[x]]

        return np.expand_dims(res_one_hot_data, axis=1)

    def _action_embedding(self, data):
        if action_mapping_action_num_mapping:
            new_mapping = action_mapping_liucun_rate_dict
            mapping_back = action_mapping_back_liucun_rate_dict
            self.mapping = new_mapping
            self.mapping_back = mapping_back

            return self._action(data)
        mapping = {}

        for x in range(len(action_list)):
            mapping[action_list[x]] = int(x)

        # integer representation
        res_one_hot_data = np.zeros_like(data)
        for x in range(len(data)):
            res_one_hot_data[x] = mapping[data[x]]

        count_action = Counter(sorted(res_one_hot_data))
        min_number = len(data) / self.action_dim

        this_i_count = 0
        action_label = 0
        new_mapping = {}
        mapping_back = {}
        action_last = -1
        for action in count_action:
            this_i_count += count_action[action]
            if this_i_count > min_number:
                for x in range(int(action_last + 1), min(int(action + 1), len(action_list))):
                    new_mapping[action_list[x]] = int(action_label)
                mapping_back[action_label] = action_list[int(action)]
                action_last = action
                action_label += 1
                this_i_count = 0

        for x in range(int(action_last + 1), len(action_list)):
            new_mapping[action_list[x]] = int(action_label)

        for action_label in range(action_label, self.action_dim):
            mapping_back[action_label] = action_list[int(action)]

        # fixme: just try to use action_mapping_liurate to encode the actions
        new_mapping = action_mapping_liucun_rate_dict
        mapping_back = action_mapping_back_liucun_rate_dict

        self.mapping = new_mapping
        self.mapping_back = mapping_back

        return self._action(data)


