import pandas as pd
from sqlalchemy import create_engine
import time
import tushare as ts
log_info={
    "username":'adim',
    'password':'Xiao15825982477#',
    'ip':'47.93.17.235',
    'database':'Index',
    'table_name': 'IndexDaily'
}
token="867c099a7df5193a7bc3b3d9f6311307a3d1885fbb39888f44754c86"
ts.set_token(token)

def get_data():
    today = time.strftime('%Y%m%d', time.localtime())
    df=ts.pro_bar(ts_code='000001.SH',trade_date=today)
    return df

def upload_data(df,username,password,ip,database,table_name):
    engine = create_engine(f'mysql+pymysql://{username}:{password}@{ip}/{database}', echo=False)
    df.to_sql(name=table_name, con=engine, if_exists='append', index=False)
    engine.dispose()

def run_script():
    df = get_data()
    upload_data(df,**log_info)
    print('Data uploaded successfully.')

# 设置定时任务，每天运行一次
while True:
    current_time = time.strftime('%H:%M:%S')
    if current_time == '00:00:00':
        run_script()
    time.sleep(1)
