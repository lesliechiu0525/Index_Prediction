class DataLoader:
    def __init__(self,token):
        self.data=None
        self._token=token #token私有
    def fetch_data(self):
        #获取时间 然后把数据从tushare导入到data属性
        pass

    def basic_plot(self): #画近期行情
        pass

    def reshape(self,*args): #根据模型reshape data格式
        pass

    def get_data(self):
        pass
        return self.data