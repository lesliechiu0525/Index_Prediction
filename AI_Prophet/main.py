import warnings
warnings.filterwarnings('ignore')
from Origin.log import Log_System,Bridge
from Origin.Data import DataLoader
from Origin.ModelSimple import ModelUniverse
log_info={
    "username":'adim',
    'password':'Xiao15825982477#',
    'ip':'47.93.17.235',
    'database':'Account'
}
param={
    'EPOCH_NUM': 1000,
    'LR': 0.01
}
import time
if __name__=='__main__':
    dataloader=DataLoader()
    info=log_info.copy()
    info['database']='IndexDatabase'
    index_bridge=Bridge()
    index_bridge.log(**info)
    dataloader.set_bridge(index_bridge)
    df=dataloader.fetch_data()
    net,train_loss,pred = ModelUniverse(df['return'],'LSTM',param)
    print(pred)