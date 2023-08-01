'''
run preditor will return all the predicted action for tickers in watchlist 
'''
import datetime
import pandas as pd
from data import MODEL_PATH, DataStorage
from preprocess import Preprocessor 
from retrying import retry
import pysnooper
import globals

MODEL_PATH = ""

from .bandwagon import Bandwagon

class Predictor:

    def __init__(self,file_name:str):
        ...
        self.ds = DataStorage()
        # self.end_time = datetime.datetime.now().strftime('%Y-%m-%d')
        df = pd.read_excel("000016closeweight(5).xls", dtype={'code':'str'}, header = 0)
        self.bw = Bandwagon(df)

    def predict(self, state):
        action = 1 if self.bw.vote() > 40 else -1 # 总分>40 买入，<40 卖出
        self.ds.save_action()   # 保存今日的action，以备后查
        return action 

    def latest_actions(self)->list[tuple]:
        ''' pretty much the same as 'watch(·)'
            w.r.t. each ticker in watchlist, get the trend(t). latest action is the last row of the dataframe
            this func can also be replaced by:
                result = [(self.end_time, t, t.iloc[-1].action) for t in self.trends(WatchList)]
                df = pd.DataFrame(result,columns=['date','ticker','action'],dtype=int)
        '''
        latest_action = lambda t: self.trend(t).iloc[-1].action
        result = [(self.end_time, t,latest_action(t)) for t in watchlist]
        df = pd.DataFrame(result,columns=['date','ticker','action'],dtype=int)
        self.ds.save_predicted(df[df.action.isin([-1,1])], if_exists = 'append') # save only action in [-1,1]
        return result # or, df as 'st.table(df)' in visualize.py
    
    def save_action(self, a, price):
        '''将本次决策保存在predicted
        a - 决策
        price - 当前close价
        '''
        pass

'''
buy or sell sz50etf by predicting its constituent
'''

if __name__ == "__main__":
    predictor = Predictor(MODEL_PATH)
