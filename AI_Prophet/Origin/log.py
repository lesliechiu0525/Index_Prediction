''' 用于用户注册和登录的系统'''
import warnings
import pandas as pd
import numpy as np
import hashlib
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

class Log_System:
    def __init__(self,size):
        self.size = size
        self.slots = [None]*size
        self.values = [None]*size

    def load(self):
        import pandas as pd
        df=pd.read_csv("user.csv")#从数据库导入数据到此instance
        self.slots=df["slots"].to_list()
        self.values=df['values'].to_list()
        #这里数据库还没有做 可以做一个excel或者csv暂时代替数据库
        return df

    def create(self,*args):#新建账号
        df=self.load()
        newusername,newuserpassword=args
        userindex=""
        userindex=int(np.exp(sum([ord(i) for i in newusername])))%self.size
        temp1=hashlib.md5()
        temp1.update(newuserpassword.encode('utf-8'))
        userpassword=temp1.hexdigest()
        if not self.slots[userindex]:
            print("Username already used.Try another one")
        else:
            self.slots[userindex]=userindex
            self.values[userindex]=userpassword#用户名与密码混淆完成
            df["slots"][userindex]=self.slots[userindex]
            df["values"][userindex]=self.values[userindex]
            df.to_csv("user.csv",index=False)
    #存入instance
        #确认后保存至本地数据库

    def login(self,*args):#登录
        import numpy as np
        username,password=args
        userindex=int(np.exp(sum([ord(i) for i in username])))%self.size
        df=self.load()
        import hashlib
        temp1=hashlib.md5()
        temp1.update(password.encode('utf-8'))
        password=temp1.hexdigest()
        if df["values"][userindex]!=password:
            print("Wrong Password or not signed up.")
            return False
        else:
            print("Logged in.")
            return True
        #先check username在不在slots 再check password的hashvalue与该索引位置\
        #的values相不相等

    def update(self,*args): #修改用户名和密码
        import numpy as np
        username,newpassword,password=args
        df=self.load()
        userindex=int(np.exp(sum([ord(i) for i in username])))%self.size
        import hashlib
        temp1=hashlib.md5()
        temp1.update(password.encode('utf-8'))
        password=temp1.hexdigest()
        temp2=hashlib.md5()
        temp2.update(newpassword.encode('utf-8'))
        newpassword=temp2.hexdigest()
        if userindex in df["slots"]:
            if password==df["values"][userindex]:
                df["values"][userindex]==newpassword
                df.to_csv("user.csv",index=False)
            else:
                print("Wrong password.")
        else:
            print("User not signed up.")
    def delete(self,*args):
        import numpy as np
        username,password=args
        userindex=int(np.exp(sum([ord(i) for i in username])))%self.size
        df=self.load()
        import hashlib
        temp1=hashlib.md5()
        temp1.update(password.encode('utf-8'))
        password=temp1.hexdigest()
        if df["values"][userindex]!=password:
            print("Wrong password. Access Denied.")
        else:
            a=input("Do you really want to delete account? If yes, press Y, else, press other keys.")
            if a=="Y":
                df["values"][userindex]=None
                df["slots"][userindex]=None
                df.to_csv("user.csv",index=False)
            else:
                print("Process aborted.")