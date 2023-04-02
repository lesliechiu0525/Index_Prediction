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
    def get(self,frequency='M'):
        data = self.data.copy()
        data = data.resample(frequency).prod()
        data = data.values
        if self.model_type == 'Linear':
            pass
        else:
            x,y = np.ones((data.shape[0]-12,12)),np.ones((data.shape[0]-12,))
            for idx in range(data.shape[0]-11):
                if idx != data.shape[0]-12:
                    x[idx,:] = data[idx:idx+12]
                    y[idx] = data[idx+12]
                else:
                    x_val = data[idx:idx+12]
            x, y, x_val = torch.from_numpy(x).to(torch.float32), \
                          torch.from_numpy(y).to(torch.float32), \
                          torch.from_numpy(x_val).to(torch.float32),
            x_val = torch.unsqueeze(x_val,dim=1)
            x,x_val = torch.unsqueeze(x,dim=2),torch.unsqueeze(x_val,dim=2)
            return x,y,x_val

class LSTM(nn.Module):
    def __init__(self):
        super(LSTM,self).__init__()
        self.input_size  = 1
        self.hidden_size = 12
        self.layers = 1
        self.lstm = nn.LSTM(self.input_size,self.hidden_size,self.layers)
        self.fc = nn.Linear(self.hidden_size,1)

    def forward(self,X):
        out,_ = self.lstm(X)
        return self.fc(out)

class Model:
    def __init__(self):
        self.instance = None
        self.model_type = None
        self.data = None
        self.dictionary = {
            'Linear': None,
            'MLP' : None,
            'RNN': None,
            'GRU': None,
            'LSTM': LSTM
        }

    def set(self,model_type,data):
        self.model_type = model_type
        self.instance = self.dictionary[self.model_type]
        self.data = data

    def train(self,EPOCH_NUM,LR):
        dataloader = DataTransform()
        dataloader.set(self.model_type,self.data)
        x,y,x_val = dataloader.get()
        loss = nn.MSELoss(reduction='none')
        net = self.instance()
        opt = torch.optim.Adam(net.parameters(),LR)
        train_loss = list()
        for epoch in range(EPOCH_NUM):
            l = loss(net(x),y).sum()
            train_loss.append(l.item())
            opt.zero_grad()
            l.backward()
            opt.step()
            if epoch%100 == 0:
                print(f'epoch:{epoch+1},loss:{train_loss[-1]:f}')
        self.instance = net
        pred = net(x_val)[-1]
        return net,train_loss,pred.item()

def ModelUniverse(data,model_type,param):
    model = Model()
    model.set(model_type,data)
    net,train_loss,pred=model.train(**param)
    return net,train_loss,pred