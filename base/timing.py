#固定标的资产择时 回测程序
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
class strategy():
    def __init__(self):
        self.signal=None
        #这两个部分后面来开发
        self.limit=None
        self.position=None
        self.factor_name=None
        self.threshold=None
        self.model=None
    def set_strategy(self,model,factor_name,position=None,hedge=False):
        self.model=model
        self.factor_name=factor_name
        self.position=position
        self.hedge=hedge

class timing_backtest():
    def __init__(self):
        self.account=pd.DataFrame(columns=["trade_date","cash","weight","value"])
        self.data=None
    def set_account(self,cash,position=None):
        self.account.loc[0]=["day0",cash,[0,0],cash]
    def fit_data(self,data):
        self.data=data
    def run(self,strategy):
        account=self.account
        df=self.data
        model=strategy.model
        factors=strategy.factor_name
        threshold=strategy.threshold
        #加载信号
        df=df.copy()
        df['single']=model.predict(df[strategy.factor_name])
        df['long-prob']=model.predict_proba(df[strategy.factor_name])[:,1]
        df['short-prob']=model.predict_proba(df[strategy.factor_name])[:,0]
        #开始交易  每日的交易信息添加到account表上 这里其实可以设置回测日期 不过并不影响 这里默认是交易input表上所有日期
        for i in range(len(df.index)):
            #常规的每日信息等于现金加上持仓乘以今天的价格记录 如果需要交易则在两个逻辑分支里面修改这几个变量
            today_info=df.loc[df.index[i]]
            pre_account=account.loc[i]
            cash=pre_account["cash"]
            weight=pre_account["weight"]
            today_price=today_info['close']
            if strategy.hedge:
                #平掉已有仓位
                print("当前价格:",today_price)
                cash_in=weight[0]*today_price+weight[1]*today_price
                print("平仓现金流入:",cash_in)
                cash_use=cash+cash_in
                print("目前可用现金:",cash_use)
                cash_long=cash_use*today_info['long-prob']
                cash_short=cash_use*today_info['short-prob']
                #按照预测概率减仓
                weight=[cash_long/today_price,-(cash_short/today_price)]
                cash=cash_use+cash_short-cash_long
                value=today_price*(sum(weight))+cash
                #将每日情况记录
                today_account = [df.index[i], cash, weight, value]
                account.loc[len(account.index)]=today_account
            else:
                print("当前价格:",today_price)
                cash_in=weight[0]*today_price+weight[1]*today_price
                print("平仓现金流入:",cash_in)
                cash_use=cash+cash_in
                print("目前可用现金:",cash_use)
                if today_info['single']==1:
                    cash_long=cash_use*today_info['long-prob']
                    if strategy.position!=None:
                        cash_long=cash_use*strategy.position
                    cash_short=0
                #只做多
                else:
                    cash_short=cash_use*today_info['short-prob']
                    if strategy.position!=None:
                        cash_short=cash_use*strategy.position
                    cash_long=0
                #按照预测概率建仓
                weight=[cash_long/today_price,-(cash_short/today_price)]
                cash=cash_use+cash_short-cash_long
                value=today_price*(sum(weight))+cash
                #将每日情况记录
                today_account = [df.index[i], cash, weight, value]
                account.loc[len(account.index)]=today_account

        #打印基础信息 这里方法将交易信息都存储到了account属性中 后面具体详细分析 可以加在analysis方法里面
        print("---------"*5,"result","---------"*5)
        print("start_value:",account.loc[0,"value"])
        print("end_value:",account.loc[max(account.index),"value"])
        print("total_return",account.loc[max(account.index),"value"]/account.loc[0,"value"])
        self.account=account
    def analysis(self):
        result_data=self.account.iloc[1:,:]
        result_data["return"]=result_data["value"]/result_data.loc[1,"value"]
        result_data.index=result_data["trade_date"]
        #取年化收益
        df_annual0=result_data["return"].resample("Y").ohlc()["close"]
        df_annual=df_annual0/df_annual0.shift(1)
        df_annual.fillna(df_annual0.iloc[0],inplace=True)
        print("-------年化收益-------")
        year_counts=len(df_annual.index)
        for i in range(year_counts):
            print(df_annual.index[i].strftime("%Y"), "{:.2%}".format(df_annual[i]-1))
        print("平均年化收益率{:.2%}".format(np.power(result_data.loc[max(result_data.index),"return"],1/year_counts)-1))
        print("-----夏普比率-----")
        print("夏普比率为{}".format((df_annual.values.mean()-1)/df_annual.values.std()))
        df_result=self.account
        fig=plt.figure(figsize=(18,12))
        plt.plot(pd.to_datetime(df_result["trade_date"][1:]),df_result["value"][1:]/df_result.loc[0,"value"],label="strategy_value")
        plt.plot(pd.to_datetime(self.data.index),self.data["close"],label="index_value")
        plt.yscale('log')
        plt.title("strategy_performance")
        plt.legend(fancybox=True,shadow=True)
        columns=["index","strategy","excess"]
        rows=["final_return"]
        market_return=self.data.loc[max(self.data.index),"close"]/self.data.loc[min(self.data.index),"close"]
        strategy_return=df_result.loc[max(df_result.index),"value"]/df_result.loc[0,"value"]
        excessive_return=strategy_return-market_return
        values=[[market_return,strategy_return,excessive_return]]
        plt.table(cellText=values
                  ,cellLoc="center"
                  ,colLabels=columns
                  ,rowLabels=rows
                  ,colWidths=[0.1]*3
                  ,rowLoc="center"
                  ,loc="lower right")
        plt.show()