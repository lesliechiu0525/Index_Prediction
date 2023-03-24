import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import hashlib

'''用于与服务器账户表连接'''
class Bridge:
    def __init__(self):
        self.engine = None
        self.status = False

    def log(self, username, password, ip, database):
        engine = create_engine(f'mysql+pymysql://{username}:{password}@{ip}/{database}', echo=False)
        self.engine = engine
        print('The bridge is open')
        self.status = True

    def download(self, table_name):
        df = pd.read_sql_table(table_name, self.engine)
        return df

    def upload(self, table_name, new_table):
        new_table.to_sql(table_name, self.engine, if_exists='replace', index=False)

    def exit(self):
        self.engine.dispose()
        print('The bridge is down')
        self.status = False

'''本地化登录系统'''
class Log_System:
    def __init__(self):
        self.size = None
        self.slots = None
        self.values = None
        self.index = None

    def load(self, df):
        self.size = df.shape[0]
        self.slots = list(df['user'])
        self.values = list(df['password_hash'])

    def HashFunction(self, x):  # 用于search
        hashvalue = int(sum([np.exp(ord(i)) for i in x])) % self.size
        return hashvalue

    def create(self, *args):  # 新建账号
        user, password = str(args[0]), str(args[0]).encode()
        hash_user = self.HashFunction(user)
        while self.slots[hash_user] != 1:
            user = input('该用户名已经存在,请重新输入(密码不需要重新输入):')
            hash_user = self.HashFunction(user)
        print('该用户名可用,创建成功')
        self.slots[hash_user] = user
        self.values[hash_user] = hashlib.sha256(password).hexdigest()

    def login(self, *args):  # 登录
        user, password = str(args[0]), str(args[1]).encode()
        hash_user = self.HashFunction(user)
        if self.slots[hash_user] == 1:
            print('用户名不存在，登录失败')
        else:
            if hashlib.sha256(password).hexdigest() == self.values[hash_user]:
                print('密码错误,登录失败')
            else:
                print('登录成功')
                self.index = hash_user

    def update(self, new_password):  # 修改用户名和密码
        try:
            self.values[self.index] = hashlib.sha256(new_password).hexdigest()
        except:
            raise IndexError('修改失败,请先登录')

    def delete(self, *args):  # 用于管理员管理账号 暂时不需要
        pass