from sklearn.metrics import roc_auc_score
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
seq_length =12
tf.random.set_seed(7)
class lstm_cv:
    def __init__(self):
        self.x_data=None
        self.y_data=None
        self.train_list=[]
        self.test_list=[]
        self.model_list=[]
        self.param_list=[]
        self.auc_list=[]
    def fit_data(self,x_data,y_data):
        self.x_data=x_data
        self.y_data=y_data
    def data_split(self,n_split=5,test_size=0.3,method="fixed"):
        x_data=self.x_data
        y_data=self.y_data
        steps=y_data.shape[0]//n_split
        if method=="fixed":
            for i in range(0,y_data.shape[0]-steps,steps):
                x_data_cross=x_data[i:i+steps]
                y_data_cross=y_data[i:i+steps]
                train_threshold=int(y_data_cross.shape[0]*(1-test_size))
                x_train_cross=x_data_cross[:train_threshold]
                y_train_cross=y_data_cross[:train_threshold]
                x_test_cross=x_data_cross[train_threshold:]
                y_test_cross=y_data_cross[train_threshold:]
                print(x_train_cross.shape,x_test_cross.shape)
                self.train_list.append([x_train_cross,y_train_cross])
                self.test_list.append([x_test_cross,y_test_cross])
        else:
            for i in range(0,y_data.shape[0],steps):
                x_data_cross=x_data[:i+steps]
                y_data_cross=y_data[:i+steps]
                train_threshold=int(len(x_data_cross)*(1-test_size))
                x_train_cross=x_data_cross[:train_threshold]
                y_train_cross=y_data_cross[:train_threshold]
                x_test_cross=x_data_cross[train_threshold:]
                y_test_cross=y_data_cross[train_threshold:]
                self.train_list.append([x_train_cross,y_train_cross])
                self.test_list.append([x_test_cross,y_test_cross])
    def cv_train(self):
        for i in range(len(self.train_list)):
            print("第{}次训练".format(i))
            train_data,test_data=self.train_list[i],self.test_list[i]
            x_train,y_train=train_data[0],train_data[1]
            x_test,y_test=test_data[0],test_data[1]
            model = Sequential()
            model.add(LSTM(32, input_shape=(seq_length,self.x_data.shape[2])))
            model.add(Dense(2))
            model.add(Activation('softmax'))
            optimizer=Adam(learning_rate=0.01)
            model.compile(loss='categorical_crossentropy', optimizer=optimizer)
            history = model.fit(x_train, y_train,epochs=30,\
                                batch_size=32,use_multiprocessing=-1,\
                               validation_data=(x_test,y_test))
            predictions_train=model.predict(x_train)
            predictions_test=model.predict(x_test)
            train_score=roc_auc_score(y_train,predictions_train[:,0].reshape(-1, 1))
            test_score=roc_auc_score(y_test,predictions_test[:,0].reshape(-1, 1))
            self.auc_list.append([train_score,test_score])
            self.model_list.append(model)
        self.auc_list=np.array(self.auc_list)
        print("cv训练完毕")
    def cv_val(self):
        print("train_auc_list",self.auc_list[:,0])
        print("test_auc_list",self.auc_list[:,1])
        train_auc=self.auc_list[:,0].mean()
        test_auc=self.auc_list[:,1].mean()
        print("train_auc:",train_auc,"test_auc:",test_auc)
    def predict(self,x):
        res0=self.predict_proba(self.x_data)
        func=lambda x:0 if x[0]>0.5 else 1
        res=np.apply_along_axis(func,axis=1,arr=res0)
        return res
    def predict_proba(self,x):
        m=len(self.model_list)
        n=x.shape[0]
        res=np.zeros((m,n))
        for i in range(m):
            res[i]=self.model_list[i].predict(self.x_data)[:,1]
        return np.array(list(zip(res.mean(axis=0),1-res.mean(axis=0))))