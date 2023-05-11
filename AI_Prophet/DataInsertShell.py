'''DatInsertShell for update my project mysql with the tushare datbases'''
from Origin.log import Bridge
import time
import tushare as ts
log_info={
    'your mysql info'
}
token= 'your tushare token'
ts.set_token(token)
pro = ts.pro_api()

def get_data():
    today = time.strftime('%Y%m%d', time.localtime())
    df = pro.index_daily(ts_code='000001.SH', start_date='20000105', end_date=today)
    return df
def upload(df):
    bridge = Bridge()
    bridge.log(**log_info)
    bridge.upload(table_name='IndexDaily',new_table=df)
    bridge.exit()

def run_script():
    df = get_data()
    upload(df)
    print('Data uploaded successfully.')

while True:
    current_time = time.strftime('%H:%M:%S')
    if current_time == '00:00:00':
        run_script()
    time.sleep(1)
