'''Create ModelUniverse'''
import torch
from torch import nn
import pandas as pd
from pylab import plt, mpl
import numpy as np
plt.style.use("seaborn")
mpl.rcParams['font.family'] = 'serif'

'''数据转化'''
class DataTransform:
    def __init__(self):
        self.model_type = None
        self.data = None
    def set(self,model_type,data):
        self.model_type = model_type
        self.data = data
    def get(self,frequency='M',factors=['Return']):
        data = self.data.copy()
        res = pd.DataFrame(columns=['Return','Volatility','Volume'])
        res['Return']= data['return'].resample(frequency).prod()
        res['Volatility'] = data['return'].resample(frequency).std()
        res['Volume'] = data['vol'].resample(frequency).prod()
        res = res[factors].values
        x_min = np.min(res, axis=0)
        x_max = np.max(res, axis=0)
        res_norm = (res - x_min) / (x_max - x_min)
        if self.model_type == 'Linear':
            pass
        else:
            dim0 = res.shape[0]-12
            dim1 = 12
            dim2 = len(factors)

            x,y = np.ones((dim0,dim1,dim2)),np.ones((dim0,1,dim2))
            for idx in range(res_norm.shape[0]-11):
                if idx != res_norm.shape[0]-12:
                    x[idx] = res_norm[idx:idx+12]
                    y[idx] = res_norm[idx+12]
                else:
                    x_val = res_norm[idx:idx+12]
            x, y, x_val = torch.from_numpy(x).to(torch.float32), \
                          torch.from_numpy(y).to(torch.float32), \
                          torch.from_numpy(x_val).to(torch.float32),
            x_val = torch.unsqueeze(x_val,dim=0)
            return x,y,x_val,x_min,x_max,res,res_norm
'''编写各类备选time series model via neural networks'''
class LSTM(nn.Module):
    def __init__(self,input_size,output_size,num_layers):
        super(LSTM,self).__init__()
        self.input_size  = input_size
        self.hidden_size = 12
        self.num_layers = num_layers
        self.lstm = nn.LSTM(self.input_size,self.hidden_size,num_layers)
        self.fc = nn.Linear(self.hidden_size,output_size)

    def forward(self,X):
        out,_ = self.lstm(X)
        out = self.fc(out[:,-1,:])
        return out

class RNN(nn.Module):
    def __init__(self, input_size, output_size,num_layers):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = 12
        self.num_layers = num_layers
        self.rnn = nn.RNN(self.input_size, self.hidden_size,num_layers=num_layers)
        self.fc = nn.Linear(self.hidden_size, output_size)

    def forward(self, X):
        out, _ = self.rnn(X)
        out = self.fc(out[:, -1, :])
        return out
class GRU(nn.Module):
    def __init__(self, input_size, output_size,num_layers):
        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = 12
        self.num_layers = num_layers
        self.gru = nn.GRU(self.input_size, self.hidden_size,num_layers=num_layers)
        self.fc = nn.Linear(self.hidden_size, output_size)

    def forward(self, X):
        out, _ = self.gru(X)
        out = self.fc(out[:, -1, :])
        return out

'''残差连接LSTM'''
class ResidualLSTM(nn.Module):
    def __init__(self, input_size, output_size,num_layers,dropout=0.1):
        super(ResidualLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = 12
        self.dropout = dropout
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=self.hidden_size,\
                           dropout=dropout,num_layers=self.num_layers)
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.dropout_layer = nn.Dropout(dropout)
        self.fc = nn.Linear(self.hidden_size,output_size)

    def forward(self, x):
        out,_ = self.lstm(x)
        residual = out
        out = self.layer_norm(out)
        out = self.dropout_layer(out)
        out = out + residual
        out = self.fc(out[:,-1,:])
        return out
'''Transformer'''
'''PositionalEncoding by TorchSource according to Attention(2017)'''
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

'''create a simple sturcture Transformer for time MultiVariables time series'''
class TransformerTimeSeries(nn.Module):
    def __init__(self, input_size, output_size,num_layers):
        super(TransformerTimeSeries, self).__init__()
        self.d_model = 64
        self.nhead = 4
        self.num_layers = num_layers
        self.dropout = 0.1
        self.input_fc = nn.Linear(input_size, self.d_model)
        self.pos_encoding = PositionalEncoding(self.d_model)
        encoder_layer = nn.TransformerEncoderLayer(self.d_model, self.nhead, dropout=self.dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, self.num_layers)
        self.out_fc = nn.Linear(self.d_model, output_size)

    def forward(self, x):
        x = self.input_fc(x)
        x = x.permute(1, 0, 2)
        x = self.pos_encoding(x)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)
        out = self.out_fc(x[:, -1, :])
        return out

'''Model for train and evaluate with any model type input'''
class Model:
    def __init__(self):
        self.instance = None
        self.model_type = None
        self.data = None
        self.Layers = None
        self.EPOCH_NUM = None
        self.LR = None
        self.FACTORS = None
        self.dictionary = {
            'RNN': RNN,
            'GRU': GRU,
            'LSTM': LSTM,
            'ResidualLSTM':ResidualLSTM,
            'TransformerTimeSeries':TransformerTimeSeries
        }

    def set(self,model_type,data):
        self.model_type = model_type
        self.instance = self.dictionary[self.model_type]
        self.data = data
    '''fit the parameters'''
    def train(self,Layers,EPOCH_NUM,LR,FACTORS):
        self.Layers = Layers
        self.EPOCH_NUM = EPOCH_NUM
        self.LR = LR
        self.FACTORS = FACTORS
        dataloader = DataTransform()
        dataloader.set(self.model_type,self.data)
        x,y,x_val,x_min,x_max,res,res_norm= dataloader.get(factors=FACTORS)
        loss = nn.MSELoss(reduction='none')
        net = self.instance(input_size=x.size(-1),\
                            output_size=x.size(-1),num_layers = Layers)
        opt = torch.optim.Adam(net.parameters(),LR)
        train_loss = list()
        for epoch in range(EPOCH_NUM):
            y_pred_flat = net(x).view(1, len(y)*len(FACTORS))
            y_true_flat = y.view(1, len(y)*len(FACTORS))
            l = loss(y_pred_flat,y_true_flat).mean()
            train_loss.append(l.item())
            opt.zero_grad()
            l.backward()
            opt.step()
            if epoch%100 == 0:
                print(f'epoch:{epoch+1},loss:{train_loss[-1]:f}')
        pred = net(x_val)[-1].detach().numpy()
        pred = pred * (x_max - x_min) + x_min #预测期
        pred_all = net(x).detach().numpy() #全期预测
        pred_all = pred_all * (x_max - x_min) + x_min
        return net,train_loss,pred,res,pred_all
    '''EVA'''
    def KFold_Eva(self,k=3):
        '''inherit attributes'''
        Layers = self.Layers
        EPOCH_NUM = self.EPOCH_NUM
        LR = self.LR
        FACTORS = self.FACTORS
        dataloader = DataTransform()
        dataloader.set(self.model_type, self.data)
        x, y, x_val, x_min, x_max, res, res_norm = \
            dataloader.get(factors=self.FACTORS)
        length = x.size(0)
        step = length // k
        mseloss_alist = list()
        for i in range(k):
            '''k fold data'''
            x_fold, y_fold = x[i * step:(i + 1) * step,:,:], y[i * step:(i + 1) * step,:,:]
            net = self.instance(input_size=x_fold.size(-1), \
                                output_size=x_fold.size(-1), num_layers=Layers)
            loss = nn.MSELoss(reduction='none')
            opt = torch.optim.Adam(net.parameters(), LR)
            train_loss = 0.0
            valid_loss = 0.0
            '''train valid split'''
            train_size = 0.8
            threshold = int(x_fold.size(0)*train_size)
            x_train,x_valid,y_train,y_valid = x_fold[:threshold,:,:],x_fold[threshold:,:,:],\
            y_fold[:threshold,:,:],y_fold[threshold:,:,:]
            for epoch in range(EPOCH_NUM):
                y_pred_flat = net(x_train).view(1, len(y_train) * len(FACTORS))
                y_true_flat = y_train.view(1, len(y_train) * len(FACTORS))
                l = loss(y_pred_flat, y_true_flat).mean()
                opt.zero_grad()
                l.backward()
                opt.step()
            '''calculate train loss and valid loss'''
            train_loss += l.item()
            y_pred_valid = net(x_valid).view(1, len(y_valid) * len(FACTORS))
            y_true_valid = y_valid.view(1, len(y_valid) * len(FACTORS))
            l_valid = loss(y_pred_valid, y_true_valid).mean()
            valid_loss += l_valid.item()
            mseloss_alist.append((train_loss, valid_loss))
        '''average all fold loss'''
        string_train = f'train-mse average:{sum([i[0] for i in mseloss_alist]) / len(mseloss_alist)}'
        string_valid = f'valid-mse average:{sum([i[1] for i in mseloss_alist])/len(mseloss_alist)}'
        return string_train +' , '+ string_valid

'''Finally we get the powerful ModelUniverse'''
'''ModelUniverse 是在我的架构下模型代码终点对象 他将所有模型函数集成到func方法'''
'''并且通过两个Link属性可以通过func方法向其他两个部分的实例传递信息'''
class ModelUniverse:
    def __init__(self):
        self.model = None
        self.data = None
        self.model_type = None
        self.result = None
        self.LinkAssistant = None
        self.LinkStrategy = None
    def set(self,data):
        self.data = data
    def func(self,FACTORS,model_type,Layers,EPOCH_NUM,LR,Evaluation):
        data = self.data.copy()
        self.model = Model()
        self.model.set(model_type,data)
        net,train_loss,pred,res,pred_all= self.model.train(Layers,EPOCH_NUM,LR,FACTORS)
        pred = list(pred)
        fig = plt.figure()
        plt.title('Loss Figure')
        plt.xlabel('EPOCH')
        plt.ylabel('LOSS')
        plt.yscale('log')
        plt.plot(range(len(train_loss)), train_loss)
        WordList=['将呈现上涨趋势','将保持稳定','下行压力增加']
        basic = np.mean(res[-12:,:],axis=0)
        reply = lambda x:WordList[0] if x>1.2 else WordList[2] if x<0.8 else WordList[1]
        string = "根据本次模型训练结果，🤖AI_Prophet预期"
        for idx in range(len(pred)):
            string+=f',{FACTORS[idx]}:{reply(pred[idx]/basic[idx])}'
        self.result = string
        if self.LinkAssistant:
            self.LinkAssistant.set(string)
        if self.LinkStrategy:
            pred_return = pred_all[:,FACTORS.index('Return')]
            self.LinkStrategy.set(pred_return)
        '''Eva'''
        evaluation = ''
        if Evaluation:
            print('hello')
            evaluation += self.model.KFold_Eva()
        return fig,string,evaluation
