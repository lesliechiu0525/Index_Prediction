class Model:
    def __init__(self,type): #type hedge single_side all-in
        self.model_list=[] #用于可视化暂时的滚动回归test
        self.type=type
        self.data=None
        self.factor_names=None
        self.limit=False

    def set(self,data,factor_names):
        self.data=data
        self.factor_names=factor_names
        pass

    def rolling_test(self,windows,step):
        pass

    def predict(self):
        pass