import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
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

'''数据处理'''
class DataTransform:
    def __init__(self):
        self.data = None
        self.transformer=None
        self.factor_names=None
    def fit(self,data):
        data = data["return"].resample("M").prod().to_frame()
        data.columns = ["return"]
        data['type'] = data['return'].apply(lambda x: 1 if x > 1 else 0)
        self.data = data

    def feature_engineering(self,type):
        data=self.data.copy()
        lag_list = [1, 2, 4, 13, 14, 18]
        for i in lag_list:
            data["lag-" + "{}".format(i)] = data.shift(i)["return"]
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
                    data["return"].rolling(i,closed="left").std()
                data["rollings_prod" + "{}".format(i)] = \
                    data["return"].rolling(i,closed="left").apply(np.prod)
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
                data["lag-" + "{}".format(i)] = data.shift(i)["return"]
            # 传统趋势和波动
            for i in range(2, 13):
                data["rollings_std" + "{}".format(i)] = \
                    data["return"].rolling(i, closed="left").var()
                data["rollings_prod" + "{}".format(i)] = \
                    data["return"].rolling(i, closed="left").apply(np.prod)
            # 加入高阶矩
            for i in range(2, 13):
                for j in range(3, 13):
                    data["rollings" + str(i) + "moment" + str(j)] = \
                        data["return"].rolling(i, closed="left").apply(lambda x: high_moment(x, j))
            data = data.dropna(how="any")
            factor_names = [i for i in data.columns in i not in ['return','type']]
            self.factor_names = factor_names
            x,y = data[factor_names],data['type']
            return x,y
        
        elif type=='interaction':
            data = self.data.copy()
            xtrain = data.drop(columns=["return", "type"])
            xtrain = (xtrain - xtrain.mean()) / xtrain.std()
            xtrain["type"] = data['type']
            # 特征相关性筛选尝试
            df_cr = xtrain.corr()['type'].sort_values()
            ts_p = [i for i in df_cr.head(20).index]
            ts_n = [i for i in df_cr.tail(20).index]
            ts_select = ts_p + ts_n
            ts_select.remove('type')
            xtrain = xtrain[ts_select]
            xtrain_new = feature_create(xtrain)
            xtrain_new['type'] = data['type']
            x,y= xtrain_new.drop(columns=['type']),xtrain_new['type']
    def feature_reduction(self,x,y):
        pca0 = PCA(svd_solver='auto')
        pca1 = PCA(n_components="mle")
        pca0.fit(x)
        x_reduction=pca0.transform(x)
        pca1.fit(x_reduction)
        x_reduction=pca1.transform(x_reduction)
        self.transformer=(pca0,pca1) #转换器传入instance的attribute
        return x_reduction,y




class Model:
    def __init__(self): #type hedge single_side all-in
        self.model_list=[]
        self.data=None
        self.factor_names=None
        self.limit=False

    def fit(self,x,y,type='Linear'):
        if type=='Linear':
            model=LogisticRegression()
            model.fit(x,y)
            self.model_list.append(model)
        elif type=='MLP':
            pass


    def data_process(self):
        pass


    def test(self,windows,step):
        pass

    def predict(self):
        pass

'''Neural Network'''
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
def TorchData_Transform(x,y,batch_size=64):
    x,y = torch.from_numpy(x).to(torch.float32),torch.from_numpy(y).to(torch.int64)
    train_dataset = TensorDataset(x,y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader
def Train(net,data_loader,num_epoch):
    net=net
    loss = nn.BCEWithLogitsLoss(reduction='none')
    opt = torch.optim.Adam(net.Parameters())
    train_loss = list()
    for epoch in range(num_epoch):
        train_loss_term = list()
        for x,y in data_loader:
            l=loss(net(x),y).sum()
            opt.zero_grad()
            l.backward()
            opt.step()
            train_loss_term.append(l.item())
        train_loss.append(sum(train_loss_term)/len(train_loss_term))
        print(f'epoch:{epoch+1},loss:{train_loss[-1]:f}')

'''Simple MLP'''
class MLP(nn.modules):
    def __init__(self,input_size):
        super().__init__()
        self._modules[0]=nn.Linear(input_size,128)
        self._modules[1]=nn.Linear(128,64)
        self._modules[2] = nn.Linear(128, 64)
        self._modules[3] = nn.Linear(128, 64)
        self._modules[4] = nn.Linear(128, 64)
        self.activation = nn.ReLU()

    def forward(self,x):
        for idx in range(len(self._modules)):
            block=self._modules[idx]
            x = block(x)
            if idx != 4:
                x = self.activation(x)
'''RNN'''
class RNN(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        '''define the h c parameter'''
        self.Whx = nn.Parameter(torch.randn(input_size,hidden_size))
        self.Whh = nn.Parameter(torch.randn(hidden_size,hidden_size))
        self.bh = nn.Parameter(torch.zeros(hidden_size))
        self.Wch = nn.Parameter(torch.randn(hidden_size,output_size))
        self.Wcx = nn.Parameter(torch.randn(input_size,output_size))
        self.bc = nn.Parameter(torch.zeros(output_size))

    def init_h(self,batch_size):
        return torch.zeros(batch_size,self.hidden_size,requires_grad=True)

    def forward(self,x): #many to one
        batch_size = x.size(0)
        h = self.init_h(batch_size=batch_size)
        c_out , h_out = torch.ones((batch_size,x.size(1),self.input_size))
        for t in x.size(1):
            h = torch.tanh(x[:,t,:]@self.Whx + h@self.Whh + self.bh)
            c = x[:,t,:]@self.Wcx + h@self.Wch + self.bc
            c_out[:,t,:],h_out[:,t,:] = c,h
        return c_out,h_out


'''LSTM'''
class LSTM(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        '''define gate '''

        '''forget gate'''
        self.Wfh = nn.Parameter(torch.randn(hidden_size,hidden_size))
        self.Wfx = nn.Parameter(torch.randn(input_size, hidden_size))
        self.bf = nn.Parameter(torch.zeros(hidden_size))
        '''input gate '''
        self.Wih = nn.Parameter(torch.randn(hidden_size,hidden_size))
        self.Wix = nn.Parameter(torch.randn(input_size, hidden_size))
        self.bi = nn.Parameter(torch.zeros(hidden_size))
        '''output  gate'''
        self.Woh = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.Wox = nn.Parameter(torch.randn(input_size, hidden_size))
        self.bo = nn.Parameter(torch.zeros(hidden_size))

        '''cell status'''
        self.Wch = nn.Parameter(torch.randn(hidden_size, output_size))
        self.Wcx = nn.Parameter(torch.randn(input_size, output_size))
        self.bc = nn.Parameter(torch.zeros(output_size))

    def init_h(self,batch_size):
        return torch.zeros(batch_size,self.hidden_size,requires_grad=True),\
    torch.zeros(batch_size,self.output_size,requires_grad=True)
    def forward(self,x):
        batch_size = x.size(0)
        h, c = self.init_h(batch_size)
        c_out , h_out = torch.ones((batch_size,self.hidden_size,self.input_size)), \
                        torch.ones((batch_size, self.hidden_size, self.input_size))
        for t in x.size(1):
            '''the gate value'''
            f = torch.sigmoid(x[:,t,:]@self.Wfx + h@self.Wfh + self.bf)
            i = torch.sigmoid(x[:,t,:]@self.Wix + h@self.Wih + self.bi)
            o = torch.sigmoid(x[:,t,:]@self.Wox + h@self.Woh + self.bo)
            '''new t statu update'''
            g = x[:,t,:]@self.Wcx + h@self.Wch + self.bc
            c = f*c + i*g
            h = (1-o)*h + o*c
            c_out[:,t,:],h_out[:,t,:] = c,h
