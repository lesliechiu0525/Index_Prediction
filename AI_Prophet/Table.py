# -*- coding: utf-8 -*-
# @Time    : 2023/4/15 16:14
# @Author  : Shark
# @Site    : 
# @File    : Table.py
# @Software: PyCharm
import warnings
warnings.filterwarnings('ignore')
import time
from Origin.log import Log_System,Bridge
from Origin.Data import DataLoader

log_info0={
    "username":name,
    'password':your_password,
    'ip':your_ip,
    'database':your_database_name
}
dataloader=DataLoader()
info=log_info.copy()
info['database']='IndexDatabase'
index_bridge=Bridge()
index_bridge.log(**info)
dataloader.set_bridge(index_bridge)
df = dataloader.fetch_data()
Tencent_bridge = Bridge()
Tencent_bridge.log(**log_info0)
Tencent_bridge.upload(table_name='IndexDaily',new_table=df)