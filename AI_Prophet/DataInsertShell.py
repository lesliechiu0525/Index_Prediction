'''DatInsertShell for update my project mysql with the tushare datbases'''
from Origin.log import Bridge
import time
import tushare as ts
log_info={
    "username":'adim',
    'password':'Xiao15825982477#',
    'ip':'47.93.17.235',
    'database':'Index',
}
token="867c099a7df5193a7bc3b3d9f6311307a3d1885fbb39888f44754c86"
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
