import sys
import os
sys.path.append("..")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import warnings
from utils.tools import split_into_lists

warnings.filterwarnings('ignore')

class Dataset_Custom(Dataset):
    def __init__(self, args, root_path, flag, num_lags, num_entries,
                 data_path,
                 target='close', scale=True, freq='h',
                 to_remove=[], date_col='date'):
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.target = target
        self.scale = scale
        self.freq = freq

        self.to_remove = to_remove
        self.date_col = date_col
        self.num_lags = num_lags
        self.num_entries = num_entries
        self.args = args

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        df_raw.rename(columns={self.date_col: 'date'}, inplace=True)
        df_raw.drop(self.to_remove, axis=1, inplace=True)

        keep_length = min(len(df_raw), self.num_entries)
        df_raw = df_raw[-keep_length:]

        cols = list(df_raw.columns)

        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]

        df_raw = df_raw.iloc[::self.args.seq_step]
        df_raw = df_raw.set_index(pd.RangeIndex(len(df_raw)))


        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        num_train -= self.num_lags + self.args.pred_len - 1
        assert num_train >= 0
        border1s = [0, num_train, num_train + num_vali]
        border2s = [num_train, num_train + num_vali, len(df_raw) - self.num_lags - self.args.pred_len + 1]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.scale:
            df_data = df_raw[df_raw.columns[1:]]
            train_data = df_data[border1s[0]:border2s[0] + self.num_lags]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
            df_stamp = df_raw['date']
            df_raw = pd.DataFrame(data, columns=cols + [self.target])
            df_raw['date'] = df_stamp

        for i in range(1, self.num_lags + 1):
            df_raw["lag_{}".format(i)] = df_raw[self.target].shift(i)
        df_raw[self.target] = df_raw[self.target].shift(1 - self.args.pred_len)

        df_raw = df_raw.dropna()

        df_raw = df_raw.set_index(pd.RangeIndex(len(df_raw)))

        df_raw['date'] = pd.to_datetime(df_raw.date)
        df_raw['month'] = df_raw.date.apply(lambda row: row.month, 1)
        df_raw['day'] = df_raw.date.apply(lambda row: row.day, 1)
        df_raw['weekday'] = df_raw.date.apply(lambda row: row.weekday(), 1)
        df_raw['hour'] = df_raw.date.apply(lambda row: row.hour, 1)
        df_raw.drop(['date'], axis = 1, inplace = True)

        df_raw.rename(columns={self.target: 'price'}, inplace=True)
        self.target = 'price'

        self.data_x = df_raw.drop([self.target], axis=1)
        self.data_y = df_raw[self.target]

        assert 0 <= border1 and border1 < border2
        assert border2 <= len(self.data_x)

        self.data_x = self.data_x.to_numpy()[border1:border2]
        self.data_y = self.data_y.to_numpy()[border1:border2]


    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index]

    def __len__(self):
        return len(self.data_x)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def split_tensors(self):
        self.data_x = np.array([split_into_lists(row) for row in self.data_x])

    def to_cnn(self):
        self.data_x = self.data_x[:, :self.num_lags]
        self.data_x = self.data_x[:, :, np.newaxis]

class Dataset_GBPCAD_hour(Dataset_Custom):
    def __init__(self, args, root_path, flag, num_lags, num_entries,
                 data_path='gbpcad_one_hour_202311210827.csv',
                 target='close', scale=True, freq='h',
                 to_remove=['id', 'provider', 'dayOfWeek', 'insertTimestamp', 'open', 'spread', 'usdPerPips', 'ask_volume', 'volume', 'ask_open', 'ask_low', 'ask_high', 'ask_close', 'ask_close', 'low', 'high'],
                 date_col='barTimestamp'):

        super().__init__(args, root_path, flag=flag, num_lags=num_lags, num_entries=num_entries, data_path=data_path, target=target, scale=scale, freq=freq,
                         to_remove=to_remove, date_col=date_col)

class Dataset_NUMSOLD_day(Dataset_Custom):
    def __init__(self, args, root_path, flag, num_lags, num_entries,
                 data_path='NUMSOLD-train.csv',
                 target='number_sold', scale=True, freq='d',
                 to_remove=['store','product'], date_col='Date'):

        super().__init__(args, root_path, flag=flag, num_lags=num_lags, num_entries=num_entries, data_path=data_path, target=target, scale=scale, freq=freq,
                         to_remove=to_remove, date_col=date_col)

class Dataset_ETHUSD_hour(Dataset_Custom):
    def __init__(self, args, root_path, flag, num_lags, num_entries,
                 data_path,
                 target='close', scale=True, freq='h',
                 to_remove=['id', 'provider', 'dayOfWeek', 'insertTimestamp', 'open', 'spread', 'usdPerPips', 'ask_volume', 'volume', 'ask_open', 'ask_low', 'ask_high', 'ask_close', 'low', 'high'],
                 date_col='barTimestamp'):

        super().__init__(args, root_path, flag=flag, num_lags=num_lags, num_entries=num_entries, data_path=data_path, target=target, scale=scale, freq=freq,
                         to_remove=to_remove, date_col=date_col)


class Dataset_WEATHER_day(Dataset_Custom):
    def __init__(self, args, root_path, flag, num_lags, num_entries,
                 data_path,
                 target='mean_temp', scale=True, freq='d',
                 to_remove=['cloud_cover', 'sunshine', 'global_radiation', 'max_temp', 'min_temp', 'precipitation', 'pressure', 'snow_depth'], date_col='date'):

        super().__init__(args, root_path, flag=flag, num_lags=num_lags, num_entries=num_entries, data_path=data_path, target=target, scale=scale, freq=freq,
                         to_remove=to_remove, date_col=date_col)

class Dataset_HOUSE_day(Dataset_Custom):
    def __init__(self, args, root_path, flag, num_lags, num_entries,
                 data_path,
                 target='price', scale=True, freq='d',
                 to_remove=['postcode', 'propertyType', 'bedrooms'], date_col='datesold'):

        super().__init__(args, root_path, flag=flag, num_lags=num_lags, num_entries=num_entries, data_path=data_path, target=target, scale=scale, freq=freq,
                         to_remove=to_remove, date_col=date_col)

class Dataset_US500USD_hour(Dataset_Custom):
    def __init__(self, args, root_path, flag, num_lags, num_entries,
                 data_path,
                 target='close', scale=True, freq='h',
                 to_remove=['open', 'usdPerPips', 'ask_open', 'ask_low', 'ask_high', 'ask_close', 'low', 'high'],
                 date_col='date'):

        super().__init__(args, root_path, flag=flag, num_lags=num_lags, num_entries=num_entries, data_path=data_path, target=target, scale=scale, freq=freq,
                         to_remove=to_remove, date_col=date_col)

class Dataset_ELECTR(Dataset_Custom):
    def __init__(self, args, root_path, flag, num_lags, num_entries,
                 data_path,
                 target='target', scale=True, freq='h',
                 to_remove=[], date_col='date'):

        super().__init__(args, root_path, flag=flag, num_lags=num_lags, num_entries=num_entries, data_path=data_path, target=target, scale=scale, freq=freq,
                         to_remove=to_remove, date_col=date_col)

class Dataset_AAPL(Dataset_Custom):
    def __init__(self, args, root_path, flag, num_lags, num_entries,
                 data_path,
                 target='Close', scale=True, freq='h',
                 to_remove=['Low', 'High', 'Adj Close', 'Open', 'Volume'], date_col='Date'):

        super().__init__(args, root_path, flag=flag, num_lags=num_lags, num_entries=num_entries, data_path=data_path, target=target, scale=scale, freq=freq,
                         to_remove=to_remove, date_col=date_col)
