import numpy as np
import random
import pandas as pd

from configs import metrics, model_features, min_ranges, pattern_ratios, _row, weeks


def get_range_of_ratios(_ratio):
    return list(np.arange(_ratio - 0.05 if _ratio - 0.05 >= 0 else 0, _ratio + 0.05  if _ratio + 0.05 <= 1 else 1, 0.005))


def get_final_ratio(r, p):
    return 0 if r + (r * p) < 0 else r + (r * p)


def promo_open_close_decide(row):
    close = 0
    if row['login'] < 0.76 and row['basket'] < 0.6 and row['payment_screen'] < 0.4 and row['ratios'] < 0.4:
        close = 1
    if row['ratios'] > 0.8 or row['payment_screen'] > 0.6:
        close = 1

    if row['score'] > 0.15 and row['ratios'] > 0.6:
        close = 1
    return close


def score_calculation(row):
    s1 = 0.2 * int(row['store'].split("_")[1])
    multiple = 1
    if row['day'] in list(range(1, 6)):
        if row['hour'] in list(range(18, 24)):
            multiple = 2
    else:
        if row['hour'] in list(range(18, 24)) + list(range(9, 13)):
            multiple = 2

    return (s1) * multiple


def open_close(ratios):
    ratios['score'] = ratios.apply(lambda row: score_calculation(row) ,axis=1)
    ratios['close'] = ratios.apply(lambda row: promo_open_close_decide(row) ,axis=1)
    return ratios


def ratio_calculation(metrics, ratios):
    result = 1
    for i in range(len(metrics)):
        result *= list(filter(lambda x: x[0] == metrics[i], ratios))[0][1]
    return get_range_of_ratios(result)


def weekly_updating(data, value, w):
    for f in ['ratios', 'login', 'basket', 'payment_screen']:
        data[f] = data.apply(lambda row: row[f] + (row[f] * value) if row[f] + (row[f] * value) >= 0 else row[f], axis=1)
    data['week'] = w
    return open_close(pd.DataFrame(data))


def get_week_of_updating_ratio():
    if np.random.random() > 0.3:
        return random.sample(list(np.arange(0.02, 0.04, 0.01)), 1)[0]
    else:
        return random.sample(list(np.arange(0.02, 0.04, 0.01)), 1)[0] * -1


class RandomDataGenerator:
    def __init__(self):
        self.metrics = metrics
        self.ratio_metrics = ['days', 'hours', 'stores']
        self.model_features = model_features
        self.busy_ratios = []
        self.patterns = []
        self.features = []
        self.sample = _row
        self.store_data = []
        self.store_week_data = []

    def get_busy_ratios(self):
        for f in self.model_features:
            count = 0
            for r in self.model_features[f]:
                if r != '_ratios':
                    self.busy_ratios += list(zip(self.metrics[self.ratio_metrics[count]],
                                                     [random.sample(self.model_features[f][r], 1)[0]
                                                      for i in self.metrics[self.ratio_metrics[count]]]))

                count += 1

    def get_patterns(self):
        for i in range(10):
            self.patterns.append([random.sample(pattern_ratios, 1)[0] for r in min_ranges])

    def get_day_store_hour_ratios(self, w, h, s):
        for f in self.model_features:
            self.model_features[f]['_ratios']['w_h_s_ratios'] = ratio_calculation([w, h, s], self.busy_ratios)

    def get_min_ratios(self, pattern, w, h, s):
        self.sample['day'], self.sample['hour'], self.sample['store'] = w, h, s
        for f in self.model_features:
            self.sample[f] = get_final_ratio(random.sample(self.model_features[f]['_ratios']['w_h_s_ratios'], 1)[0], pattern)
        self.store_data.append(self.sample)

    def week_of_updated_ratios(self, w):
        self.store_data = weekly_updating(pd.DataFrame(self.store_data), get_week_of_updating_ratio(), w)
        self.store_week_data += self.store_data.to_dict('results')

    def write_store_week_to_csv(self, s):
        open_close(pd.DataFrame(self.store_data)).to_csv("data/availability_ratios_" + s + ".csv", index=False)

    def read_store_week_from_csv(self, s):
        self.store_data = pd.read_csv("data/availability_ratios_" + s + ".csv")

    def calculate_week_of_availabilities(self):
        self.get_patterns()
        self.get_busy_ratios()
        for s in self.metrics[self.ratio_metrics[2]]:
            self.store_data = []
            for h in self.metrics[self.ratio_metrics[1]]:
                for w in self.metrics[self.ratio_metrics[0]]:
                    self.get_day_store_hour_ratios(w, h, s)
                    _patterns = random.sample(self.patterns, 1)[0]
                    for m in self.metrics['mins']:
                        _min_range_idx = list(map(lambda x: x[2], list(filter(lambda x: x[0] > m >= x[1], min_ranges))))[0]
                        _p = _patterns[_min_range_idx]
                        self.get_min_ratios(_p, w, h, s)
            self.write_store_week_to_csv(s)

    def generate_random_data(self):
        self.calculate_week_of_availabilities()
        for s in self.metrics[self.ratio_metrics[2]]:
            self.read_store_week_from_csv(s)
            for week in weeks:
                self.week_of_updated_ratios(week)
            self.store_data = self.store_week_data
            self.write_store_week_to_csv(s)






