import pandas as pd
import random
Wordbase = pd.read_csv('words.csv')
dictionary = None
class Assistant:
    def __init__(self):
        self.result = None
        self.dictionary = {
            '将呈现上涨趋势':'1',
            '将保持稳定':'0',
            '下行压力增加':'1'
        }
        self.Words = Wordbase
    def set(self,result):
        if result:
            str_list = result.split(',')
            str_list = str_list[1:]
            self.result = str_list
        else:
            self.result = result
    def reply(self,string):
        if '分析' in string:
            if self.result:
                base = random.choice(self.result)
                key = base.split(':')[1]
                idx = self.dictionary[key]
                if base.split(':')[0] == 'Volatility':
                    idx = str(-int(idx))
                analysis = random.choice(self.Words[idx])
                return '您建立的模型预期'+base +' , AI_Prophet建议您:'+ analysis
            else:
                return '建议您先使用AI_Prophet的Model Predict功能，\
                我再基于模型分析结果给出建议'
        elif 'A股' in string or '市场' in string:
            return '我是基于中国A股的上证综指时序数据来对市场进行判断的，\
            您可以使用我的Model Predict功能，然后再来找我回复‘分析’我能够根据模型结果给出我的建议'
        elif '推荐' in string:
            return '天机不可泄露'
        elif '使用' in string or '怎么' in string:
            return '点击左上角可以切换各个模块，选择你需要的功能，我能跟您提供帮助'
        elif '交易' in string or '策略' in string:
            return '进入Strategy界面，我可以帮你测试我提供的模型、' \
                   ',在历史上的回测效果'
        else:
            return random.choice(['想必你心中已有答案','要不说得别的，我可以帮您分析市场'])