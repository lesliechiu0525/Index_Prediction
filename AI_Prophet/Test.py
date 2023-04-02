import numpy as np
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from Origin.log import Log_System,Bridge
from Origin.Data import DataLoader
from Origin.ModelSimple import ModelUniverse
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
dataloader=DataLoader()
info=log_info.copy()
info['database']='IndexDatabase'
index_bridge=Bridge()
index_bridge.log(**info)
dataloader.set_bridge(index_bridge)
df = dataloader.fetch_data()
model_universe = ModelUniverse()
model_universe.set(df)

with gr.Blocks() as demo:
    gr.Markdown("<h1><center>AI_Prophet is All You Need In Investment</center></h1>")
    with gr.Tab("Visual Analysis"):
        choice = ['performance', 'analysis','ACF']
        method_input = gr.Radio(choices=choice)
        plot_output = gr.Plot()
        plot_button = gr.Button("Plot")
    with gr.Tab("Model Prediction"):
        choice = ['Linear','RNN','GRU','LSTM']
        Type_input = gr.Radio(choices=choice,label='ModelType')
        Epoch_input = gr.Slider(minimum=101,maximum=1001,label='EPOCH')
        LR_input = gr.Slider(minimum=0.001,maximum=0.1,label='LearningRate')
        Loss_output = gr.Plot(label='Loss Figure')
        Pre_output = gr.Text(label='Predict')
        image_button = gr.Button("Predict")

    with gr.Accordion("Open for More!"):
        gr.Markdown("Look at me...")
    plot_button.click(dataloader.basic_plot, inputs=method_input, outputs=plot_output)
    image_button.click(model_universe.func, \
                       inputs=[Type_input,Epoch_input,LR_input],\
                       outputs=[Loss_output,Pre_output])
if __name__=='__main__':
    demo.title = "AI_Prophet 🤖"
    demo.launch()
    # demo.launch(server_name="0.0.0.0", server_port=7860, \
    #              share=False,auth=('lesliechiu','0525'))