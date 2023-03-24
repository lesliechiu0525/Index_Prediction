from Origin.log import Log_System,Bridge
from Origin.Data import DataLoader
from Origin.Model import Model
log_info={
    "username":'adim',
    'password':'Xiao15825982477#',
    'ip':'47.93.17.235',
    'database':'Account'
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
    dataloader.basic_plot(method='performance')
    dataloader.basic_plot(method='analysis')
    dataloader.basic_plot(method='ACF')