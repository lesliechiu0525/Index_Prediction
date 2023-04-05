import pandas as pd
from pylab import plt, mpl
import numpy as np
plt.style.use("seaborn")
mpl.rcParams['font.family'] = 'serif'
from statsmodels.graphics.tsaplots import plot_acf
class DataLoader:
    def __init__(self):
        self.data=None
        self.bridge=None
    def set_bridge(self,bridge):
        self.bridge=bridge

    def fetch_data(self):
        data=self.bridge.download(table_name='IndexDaily').reset_index(drop=True)
        data = data.sort_values(by='trade_date',ascending=True)
        data['trade_date']=pd.to_datetime(data['trade_date'])
        data.index = data['trade_date']
        data['return'] = data['close']/data.shift(1)['close']
        data.dropna(how='any',inplace=True)
        self.data=data
        self.bridge.exit()
        return data

    def basic_plot(self,method): #画近期行情
        data=self.data.copy()
        data['MA5'] = data['close'].rolling(5).mean()
        data['MA10'] = data['close'].rolling(10).mean()
        data['MA120'] = data['close'].rolling(120).mean()
        if method=='performance':
            fig = plt.figure()
            plt.subplot(211)
            plt.title("000001SH Performance Daily")
            data['close'].plot(label='close')
            plt.subplot(212)
            plt.title('000001SH Performance Annual')
            y=data['return'].resample('Y').prod()
            x=[i.strftime('%Y')[-2:] for i in y.index]
            plt.bar(x,y-1)
            plt.subplots_adjust(hspace=1)
            # plt.show()
            return fig

        elif method=='analysis':
            fig = plt.figure()
            plt.subplot(211)
            plt.title("000001SH Momentum in 2023")
            data['2023']['close'].plot(label='close')
            data['2023']['MA5'].plot(label='MA5')
            data['2023']['MA10'].plot(label='MA10')
            data['2023']['MA120'].plot(label='MA120')
            plt.legend()
            plt.subplot(212)
            plt.title('000001SH Volatility Week in 2023')
            y=data['2023']['return'].resample('W').std()
            plt.bar(range(len(y)),y)
            plt.subplots_adjust(hspace=1)
            # plt.show()
            return fig

        elif method=='ACF':
            fig = plt.figure()
            monthly_return=data['return'].resample('M').prod()
            plot_acf(monthly_return)
            return plot_acf(monthly_return)

    def reshape(self): #根据模型reshape data格式
        data=self.data.copy()
        data.reset_index(drop=True)

        pass

    def get_data(self):
        return self.data