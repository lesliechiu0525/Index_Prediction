import tushare as ts
import time
import pandas as pd
import matplotlib.pyplot as mp
import numpy as np
class DataLoader:
    def __init__(self,token):
        self.data=None
        self.token=token #token私有
        ts.set_token(self.token)
    def fetch_data(self):
        # 获取时间 然后把数据从tushare导入到data属性
        enddate=str(time.strftime("%Y-%m-%d",time.localtime())).replace("-","")
        startdate=enddate[0:2]+str(int(enddate[2:4])-2)+enddate[4:8]
        self.data=(ts.pro_bar(ts_code='000001.SH', asset='I', start_date=startdate, end_date=enddate)).iloc[::-1,:].reset_index(drop=True)
        pass

    def basic_plot(self): #画近期行情
        temp=self.data["trade_date"].to_list()
        mp.rcParams['font.sans-serif'] = ['SimHei']
        mp.figure(dpi=40, facecolor="white", figsize=(13, 13))
        mp.subplot(311)
        mp.title("上证指数近两年走势")
        length=self.data.shape[0]
        numbers=np.arange(length)
        mp.plot(numbers,self.data['close'])
        mp.xticks(numbers[::30],(pd.to_datetime(np.array(self.data.trade_date, dtype="U8")[::30])).date,rotation=30)
        mp.subplot(312)
        mp.title("成交量")
        mp.bar(numbers,self.data['vol'],width=1)
        mp.xticks(numbers[::30], (pd.to_datetime(np.array(self.data.trade_date, dtype="U8")[::30])).date,rotation=30)
        mp.subplot(313)
        mp.title('上证指数近60交易日走势')
        mask = self.data[-60:].close > self.data[-60:].open
        colors = np.zeros(mask.size, dtype="U5")
        colors[:] = "green"
        colors[mask] = "red"
        mp.bar(np.arange(60),(self.data.close[-60:]-self.data.open[-60:]),bottom=self.data.open[-60:],color=colors,width=1)
        mp.vlines(np.arange(60),self.data.low[-60:],self.data.high[-60:],color=colors)
        mp.xticks(np.arange(60)[::5],(pd.to_datetime(np.array(self.data.trade_date[-60:], dtype="U8")[::5])).date, rotation=30)
        mp.show()
        pass

    def reshape(self,*args): #根据模型reshape data格式
        pass

    def get_data(self):
        pass
        return self.data