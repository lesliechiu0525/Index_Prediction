import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from Origin.log import Log_System,Bridge
from Origin.Data import DataLoader
# from Origin.ModelSimple import ModelUniverse
import gradio as gr
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
    df = dataloader.fetch_data()
    # net,train_loss,pred = ModelUniverse(df['return'],'LSTM',param)
    # iface =gr.Interface(ModelUniverse,)
    ifact = gr.Interface(dataloader.basic_plot,
                         inputs='text',
                         outputs='plot',
                         theme='peach')
    ifact.title = "AI_Prophet 🤖"
    # dataloader.basic_plot(method='performance')
    ifact.launch(server_name="0.0.0.0", server_port=7860, \
                 share=False,auth=('lesliechiu','0525'))
    # ifact.launch(auth=('lesliechiu','0525'))



