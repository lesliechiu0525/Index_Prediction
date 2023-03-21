from lightgbm import LGBMClassifier
import itertools
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
class ts_cv:
    def __init__(self):
        self.ts=None
        self.n_split=None
        self.model_list=[]
        self.best_param=None
        self.factor_name=None
        self.target=None
        self.n_split=None
        self.test_size=None
    def fit(self,factor_name,target):
        self.factor_name=factor_name
        self.target=target
    def cv_search(self,ts,param,n_split=5,test_size=1):
        self.n_split=n_split
        self.test_size=test_size
        self.ts=ts
        self.n_split=n_split
        dic=param
        #用于记录参数组合和分数
        res_para=[]
        res_score=[]
        #找到所有可能的参数组合
        l_value=[i for i in dic.values()]
        l_values=[]
        for i in itertools.product(*l_value):
            l_values.append([j for j in i])
        #需要计算多少次
        totle_counts=len(l_values)
        l_keys=[i for i in dic.keys()]
        counter=1
        print("一共需要计算{}组参数".format(totle_counts))
        date_totle=len(ts.index)
        step=date_totle//n_split
        train_threshold=int(step*(1-test_size))
        print(step)
        for i in range(len(l_values)):
            param=dict(zip(l_keys,l_values[i]))
            test_score_ini=[]
            param_ini=[]
            for j in range(step,date_totle-step,step):
                data_cross=ts.iloc[j-step:j,:]
                train=data_cross[:train_threshold]
                test=data_cross[train_threshold:]
                model=LGBMClassifier(**param).fit(train[self.factor_name],train[self.target])
                auc=roc_auc_score(test[self.target],model.predict_proba(test[self.factor_name])[:,1])
                test_score_ini.append(auc)
            print("搜寻进程：{}/{}".format(counter,totle_counts))
            counter+=1
            res_score.append(sum(test_score_ini)/len(test_score_ini))
            res_para.append(param)
        max_index=res_score.index(max(res_score))
        self.best_param=res_para[max_index]
        return (res_para[max_index],res_score[max_index])
        
    def cv_train(self):
        #先训练存储模型
        ts=self.ts
        n_split=self.n_split
        param=self.best_param
        date_totle=len(ts.index)
        step=date_totle//n_split
        for j in range(step,date_totle-step,step):
                train=ts.iloc[j-step:j,:]
                model=LGBMClassifier(**param).fit(train[self.factor_name],train[self.target])
                self.model_list.append(model)
    def predict(self,x):
        res0=self.predict_proba(x)
        func=lambda x:0 if x[0]>0.5 else 1
        res=np.apply_along_axis(func,axis=1,arr=res0)
        return res
    def predict_proba(self,x):
        m=len(self.model_list)
        n=x.shape[0]
        res=np.zeros((m,n))
        for i in range(m):
            res[i]=self.model_list[i].predict_proba(x)[:,0]
        return np.array(list(zip(res.mean(axis=0),1-res.mean(axis=0))))