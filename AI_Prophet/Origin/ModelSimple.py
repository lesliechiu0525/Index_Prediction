import torch
from torch import nn
import pandas as pd
from pylab import plt, mpl
import numpy as np
plt.style.use("seaborn")
mpl.rcParams['font.family'] = 'serif'
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

class LSTM(nn.Module):
    def __init__(self,input_size,output_size):
        super(LSTM,self).__init__()
        self.input_size  = input_size
        self.hidden_size = 12
        self.lstm = nn.LSTM(self.input_size,self.hidden_size)
        self.fc = nn.Linear(self.hidden_size,output_size)

    def forward(self,X):
        out,_ = self.lstm(X)
        out = self.fc(out[:,-1,:])
        return out

class RNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = 12
        self.rnn = nn.RNN(self.input_size, self.hidden_size)
        self.fc = nn.Linear(self.hidden_size, output_size)

    def forward(self, X):
        out, _ = self.rnn(X)
        out = self.fc(out[:, -1, :])
        return out
class GRU(nn.Module):
    def __init__(self, input_size, output_size):
        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = 12
        self.gru = nn.GRU(self.input_size, self.hidden_size)
        self.fc = nn.Linear(self.hidden_size, output_size)

    def forward(self, X):
        out, _ = self.gru(X)
        out = self.fc(out[:, -1, :])
        return out


class Model:
    def __init__(self):
        self.instance = None
        self.model_type = None
        self.data = None
        self.dictionary = {
            'Linear': None,
            'MLP' : None,
            'RNN': RNN,
            'GRU': GRU,
            'LSTM': LSTM
        }

    def set(self,model_type,data):
        self.model_type = model_type
        self.instance = self.dictionary[self.model_type]
        self.data = data

    def train(self,EPOCH_NUM,LR,FACTORS):
        dataloader = DataTransform()
        dataloader.set(self.model_type,self.data)
        x,y,x_val,x_min,x_max,res,res_norm= dataloader.get(factors=FACTORS)
        loss = nn.MSELoss(reduction='none')
        net = self.instance(input_size=x.size(-1),output_size=x.size(-1))
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
        self.instance = net
        pred = net(x_val)[-1].detach().numpy()
        pred = pred * (x_max - x_min) + x_min #预测期
        pred_all = net(x).detach().numpy() #全期预测
        pred_all = pred_all * (x_max - x_min) + x_min
        return net,train_loss,pred,res,pred_all
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
    def func(self,FACTORS,model_type,EPOCH_NUM,LR):
        data = self.data.copy()
        self.model = Model()
        self.model.set(model_type,self.data)
        net,train_loss,pred,res,pred_all= self.model.train(EPOCH_NUM,LR,FACTORS)
        pred = list(pred)
        fig = plt.figure()
        plt.title('Loss Figure')
        plt.xlabel('EPOCH')
        plt.ylabel('LOSS')
        plt.yscale('log')
        plt.plot(range(len(train_loss)), train_loss)
        WordList=['将呈现上涨趋势','将保持稳定','下行压力增加']
        basic = np.mean(res[-10:,:],axis=0)
        reply = lambda x:WordList[0] if x>1.2 else WordList[2] if x<0.8 else WordList[1]
        string = "根据本次模型训练结果，🤖AI_Prophet预期"
        for idx in range(len(pred)):
            string+=f',{FACTORS[idx]}:{reply(pred[idx]/basic[idx])}'
        self.model = net
        self.result = string
        if self.LinkAssistant:
            self.LinkAssistant.set(string)
        if self.LinkStrategy:
            pred_return = pred_all[:,FACTORS.index('Return')]
            self.LinkStrategy.set(pred_return)
        return fig,string