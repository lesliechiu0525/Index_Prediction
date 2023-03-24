import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
'''计算高阶矩'''
def high_moment(x, j):
    if all(x.isnull()):
        return np.nan
    else:
        x = x.values
        return np.power((x - x.mean()), j).mean()

'''特征交互'''
def feature_create(train):
    train = train.copy()
    xtrain = train
    colname = [i for i in xtrain.columns]
    for i in range(len(colname)):
        for j in range(i + 1, len(colname)):
            xtrain[colname[i] + '_sub_' + colname[j]] = xtrain.iloc[:, i] - xtrain[colname[j]]
            xtrain[colname[i] + '_add_' + colname[j]] = xtrain.iloc[:, i] + xtrain[colname[j]]
            xtrain[colname[i] + '_multi_' + colname[j]] = xtrain.iloc[:, i] * xtrain[colname[j]]
            xtrain[colname[i] + '_div_' + colname[j]] = xtrain.iloc[:, i] / (xtrain.iloc[:, j] + 1)
    return xtrain
class DataTransform:
    def __init__(self):
        self.data = None
        self.transformer=None
        self.factor_names=None
    def fit(self,data):
        data = data["return"].resample("M").prod().to_frame()
        data.columns = ["monthly_return"]
        data['type'] = data['monthly_return'].apply(lambda x: 1 if x > 1 else 0)
        self.data = data

    def feature_engineering(self,type):
        data=self.data.copy()
        lag_list = [1, 2, 4, 13, 14, 18]

        for i in lag_list:
            data["lag-" + "{}".format(i)] = data.shift(i)["monthly_return"]
        data = data.drop(how='any')
        factor_names = ['lag-4', 'lag-13', "lag-14"]
        self.factor_names=factor_names
        x, y = data[factor_names], data['type']

        if type=='Simple-Lag':
            return x,y

        elif type=='Normal':
            data=self.data.copy()
            for i in range(2, 13):
                data["rollings_std" + "{}".format(i)] = \
                    data["monthly_return"].rolling(i,closed="left").std()
                data["rollings_prod" + "{}".format(i)] = \
                    data["monthly_return"].rolling(i,closed="left").apply(np.prod)
            data = data.dropna(how="any")
            factor_names.extend(['rollings_std3', "rollings_prod6", "rollings_prod10"])
            self.factor_names = factor_names
            # 第二次建立逻辑回归模型
            x,y = data[factor_names],data['type']
            return x,y

        elif type=='High-Moment-Normal':
            data = self.data.copy()
            lag_list = [1, 2, 4, 13, 14, 18]
            for i in lag_list:
                data["lag-" + "{}".format(i)] = data.shift(i)["monthly_return"]
            # 传统趋势和波动
            for i in range(2, 13):
                data["rollings_std" + "{}".format(i)] = \
                    data["monthly_return"].rolling(i, closed="left").var()
                data["rollings_prod" + "{}".format(i)] = \
                    data["monthly_return"].rolling(i, closed="left").apply(np.prod)
            # 加入高阶矩
            for i in range(2, 13):
                for j in range(3, 13):
                    data["rollings" + str(i) + "moment" + str(j)] = \
                        data["monthly_return"].rolling(i, closed="left").apply(lambda x: high_moment(x, j))
            data = data.dropna(how="any")
            factor_names = [i for i in data.columns in i not in ['monthly_return','type']]
            self.factor_names = factor_names
            x,y = data[factor_names],data['type']
            return x,y
        
        elif type=='interaction':
            data = self.data.copy()
            xtrain = data.drop(columns=["monthly_return", "monthly_return_type"])
            xtrain = (xtrain - xtrain.mean()) / xtrain.std()
            xtrain["type"] = data['monthly_return_type']
            # 特征相关性筛选尝试
            df_cr = xtrain.corr()['monthly_return_type'].sort_values()
            ts_p = [i for i in df_cr.head(20).index]
            ts_n = [i for i in df_cr.tail(20).index]
            ts_select = ts_p + ts_n
            ts_select.remove('monthly_return_type')
            xtrain = xtrain[ts_select]
            xtrain_new = feature_create(xtrain)
            xtrain_new['monthly_return_type'] = data['monthly_return_type']
            x,y= xtrain_new.drop(columns=['monthly_return_type']),xtrain_new['monthly_return_type']



class Model:
    def __init__(self,type): #type hedge single_side all-in
        self.model_list=[] #用于可视化暂时的滚动回归test
        self.type=type
        self.data=None
        self.factor_names=None
        self.limit=False

    def fit(self,data):
        pass

    def data_process(self):
        pass


    def rolling_test(self,windows,step):
        pass

    def predict(self):
        pass