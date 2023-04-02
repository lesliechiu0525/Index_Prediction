import numpy as np
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from log import Log_System,Bridge
from Data import DataLoader
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
dataloader=DataLoader()
info=log_info.copy()
info['database']='IndexDatabase'
index_bridge=Bridge()
index_bridge.log(**info)
dataloader.set_bridge(index_bridge)
df = dataloader.fetch_data()

def flip_text(x):
    return x[::-1]


def flip_image(x):
    return np.fliplr(x)


with gr.Blocks() as demo:
    gr.Markdown("<h1><center>AI_Prophet is All You Need In Investment</center></h1>")
    with gr.Tab("Visual Analysis"):
        choice = ['performance', 'analysis','ACF']
        method_input = gr.Radio(choices=choice)
        plot_output = gr.Plot()
        plot_button = gr.Button("Plot")
    with gr.Tab("Model Prediction"):
        with gr.Row():
            image_input = gr.Image()
            image_output = gr.Image()
        image_button = gr.Button("Predict")

    with gr.Accordion("Open for More!"):
        gr.Markdown("Look at me...")
    plot_button.click(dataloader.basic_plot, inputs=method_input, outputs=plot_output)
    image_button.click(flip_image, inputs=image_input, outputs=image_output)
if __name__=='__main__':
    demo.title = "AI_Prophet 🤖"
    demo.launch(auth=('lesleichiu','0525'))