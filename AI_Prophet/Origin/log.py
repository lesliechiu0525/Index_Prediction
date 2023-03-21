''' 用于用户注册和登录的系统'''
class Log_System:
    def __init__(self,size):
        self.size = size
        self.slots = [None]*size
        self.values = [None]*size

    def HashFuction(self): #哈希函数
        pass

    def load(self):#从数据库导入数据到此instance
        #这里数据库还没有做 可以做一个excel或者csv暂时代替数据库
        pass

    def create(self,*args):#新建账号
        #存入instance
        #确认后保存至本地数据库
        pass

    def log(self,*args):#登录
        username,password=args
        #先check username在不在slots 再check password的hashvalue与该索引位置\
        #的values相不相等

    def update(self,*args): #修改用户名和密码
        pass

    def delete(self,*args): #删除账号
        pass
