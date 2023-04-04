import numpy as np
import warnings
warnings.filterwarnings('ignore')
import random
import time
import pandas as pd
from Origin.log import Log_System,Bridge
from Origin.Data import DataLoader
from Origin.ModelSimple import ModelUniverse
import gradio as gr
from Origin.Strategy import Strategy,BackTrade
from Origin.Guidence import Assistant
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
strategy = Strategy()
backtrader = BackTrade()
df = df.resample('M').last()
backtrader.fit(df, factor_names=['vol'])
backtrader.set_strategy(strategy)
assistant = Assistant()
model_universe.LinkAssistant = assistant
model_universe.LinkStrategy = strategy


with  gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("<h1><center>AI_Prophet is All You Need In Investment</center></h1>")
    with gr.Tab("Visual Analysis"):
        choice = ['performance', 'analysis','ACF']
        method_input = gr.Radio(choices=choice)
        plot_output = gr.Plot()
        plot_button = gr.Button("Plot")

    with gr.Tab("Model Prediction"):
        Factors_Input = gr.CheckboxGroup(['Return','Volatility','Volume'],label='Factors')
        choice = ['RNN','GRU','LSTM']
        Type_input = gr.Radio(choices=choice,label='ModelType')
        Epoch_input = gr.Slider(minimum=101,maximum=501,label='EPOCH')
        LR_input = gr.Slider(minimum=0.001,maximum=0.1,label='LearningRate')
        Loss_output = gr.Plot(label='Loss Figure')
        Pre_output = gr.Text(label='Prediction')
        image_button = gr.Button("Predict")
        image_button.click(model_universe.func, \
                           inputs=[Factors_Input, Type_input, \
                                   Epoch_input, LR_input], \
                           outputs=[Loss_output, Pre_output])

    with gr.Tab("Strategy"):
        gr.Markdown('<h3 style="text-align:center;font-weight:bold;">\
        !!Pay attention to Model Predict Firstly!!</h3>')
        Limit_input = gr.Checkbox(label='Limit',info='False for long only')
        Percent_input = gr.Slider(minimum=0.2,maximum=1.0,label='Percent Use')
        Performance_out = gr.Plot(label='Backtest Result')
        gr.Markdown('<h2 style="text-align:center;font-size:28px;font-family:Arial,\
         sans-serif;">Backtest Summary</h2>')
        Dataframe_out = gr.Dataframe()
        backtest_button = gr.Button('Backtest')
        backtest_button.click(backtrader.run, \
                           inputs=[Limit_input, Percent_input], \
                           outputs=[Performance_out,Dataframe_out])

    with gr.Tab('Guides'):
        chatbot = gr.Chatbot(label='AI_Prophet 🤖')
        msg = gr.Textbox(label='Question',info='How can I help you?')
        clear = gr.Button("Clear")
        def user(user_message, history):
            return "", history + [[user_message, None]]
        def bot(history):
            bot_message = assistant.reply(history[-1][0])
            history[-1][1] = bot_message
            time.sleep(1)
            return history

        msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot, chatbot, chatbot
        )
        clear.click(lambda: None, None, chatbot, queue=False)

    with gr.Accordion("Open for More!"):
        gr.Markdown("Developed by LeslieChiu of Southwestern University of Finance and Economics")
        gr.Markdown('AI_Prophet is an application developed by LeslieChiu \
        for quantitative analysis of the Chinese stock market,\
         based on  time series neural network models')
        gr.HTML('<img src="Asset.png">')
    plot_button.click(dataloader.basic_plot, inputs=method_input, outputs=plot_output)

if __name__=='__main__':
    demo.title = "AI_Prophet 🤖"
    demo.launch(auth=('lesliechiu','0525'))
    # demo.launch(server_name="0.0.0.0", server_port=7860, \
    #              share=False,auth=('lesliechiu','0525'))