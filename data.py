import baostock as bs
import os
import sqlite3
import pandas as pd
from functools import partial
from globals import MAIN_PATH, OCLHVA, MKT_OCLHVA, BAOSTOCK_MAPPING, WINDOW_SIZE, TS_TOKEN
from datetime import datetime, timedelta
from itertools import count
import tushare as ts

DATABASE = "stock.db"
DATABASE_PATH = MAIN_PATH  # 这句话在其他import的时候就已经执行，所以未必能达到想要的效果 # 直接调用上面globals.MAIN_PATH 导致不能正确建立conn，main不能及时修改globals.MAIN_PATH
ALL_TICKERS_DATA = "all_tickers"
RAW_DATA = "downloaded" # by 'date'
PROCESSED_DATA = "downloaded" # by 'date'
TRAINED_DATA = "downlowded" #  by 'date'
EVALUATED_DATA = "downloaded" # by 'date' 下一回合数据到来的时候即清空
TRAIN_HISTORY_DATA = "train_history" # by 'episode'  长期保存做评估
ACTION_HISTORY_DATA = "action_hisory" #  by 'action' action_history结构和evaluated一样，但去掉了action==0的记录，供长期保存做评估
PREDICTED_DATA = "predicted" # by 'action', action prediction for WatchList，保存用作评估
MODEL_PATH = "./model/" # table name
MODEL_NAME = "trained_model" # trained model name

class DataStorage():
    def __init__(self, path = DATABASE_PATH , database = DATABASE) -> None:
        self.conn = sqlite3.connect(os.path.join(path, database))
        if not self.conn:
            raise(f"connection failed with {path}/{database}")
        # raw, preprocessed, trained, evaluated can be unified as in one table, or two, since they are all by 'ticker,date'
        # the following wrappers follows the order of whole process
        # raw.columns = baostock.columns = ['date', 'code', 'open', 'high', 'low', 'close', 
        # 'preclose', 'volume', 'amount', 'adjustflag', 'turn', 'tradestatus', 'pctChg', 'isST']
        # preprocessed.columns += [normalized oclhva,landmarks,indicators,embeddings]
        # trained.columns = ['date','ticker','landmark','close','action','reward']
        # evaluated.columns += ['asset_pct_chg'] 
        self.save_tickers = partial(self.save,to_table = ALL_TICKERS_DATA)
        self.save_raw = partial(self.save,to_table = RAW_DATA)
        self.load_raw = partial(self.load,from_table = RAW_DATA)
        self.save_processed = partial(self.save,to_table = PROCESSED_DATA)
        self.load_processed = partial(self.load, from_table = PROCESSED_DATA)
        self.save_trained = partial(self.save,to_table = TRAINED_DATA)
        self.load_trained = partial(self.load,from_table = TRAINED_DATA)
        self.save_evaluated = partial(self.save, to_table = EVALUATED_DATA)
        self.load_evaluated = partial(self.load, from_table = EVALUATED_DATA)
        # 'train_history' is to preserve the history of training, by 'episode'
        # train_history.columns =  ['episode','ticker','train_date','mean','std','asset_change']
        self.append_train_history = partial(self.save, to_table = TRAIN_HISTORY_DATA,if_exists='append')
        self.load_train_history = partial(self.load, from_table = TRAIN_HISTORY_DATA) 
        # 'action_history' is to preserve the history of predicted action for train data, by 'action'
        # action_history.columns =  ['date','ticker','action','reward','asset_change'] 
        self.append_action_history = partial(self.save, to_table = ACTION_HISTORY_DATA,if_exists='append')
        self.load_action_history = partial(self.load, from_table = ACTION_HISTORY_DATA) 
        # 'predicted' is to preserve the history of predicted action for WatchList, by 'action'
        # predicted.columns = ['predict_date','ticker','action','asset_pct_chg']
        self.append_predicted = partial(self.save, to_table = PREDICTED_DATA,if_exists='append')
        self.load_predicted = partial(self.load, from_table = PREDICTED_DATA)

    def __del__(self) -> None:
        self.conn.close()

    def load(self,from_table:str) -> pd.DataFrame:
        return pd.read_sql(f'SELECT * FROM {from_table}', con = self.conn,index_col='index')

    def save(self, df:pd.DataFrame, to_table:str, if_exists='replace'):
        return df.to_sql(name=to_table,con = self.conn, if_exists = if_exists)

class DataWorker(object):
    def __init__(self) -> None:
        self.begin = "2000-01-01"

    def __del__(self) -> None:
        pass
    
    # @property
    def all_tickers(self) -> pd.DataFrame :
        pass
    
    def minute(self,ticker,start_date = None, end_date = None, ktype='5') -> pd.DataFrame :
        pass
    
    def latest(self,ticker, ktype = '5', days = 20):
        pass
    
    def save(self, rs:pd.DataFrame) -> bool:
        # self.ds = DataStorage()
        pass
    
    def market_calendar(self):
        pass

    def market(self,ticker) -> str:
        ''' returns the market index of designated ticker. 
        http://baostock.com/baostock/index.php/指数数据
        综合指数，例如：sh.000001 上证指数，sz.399106 深证综指 等；
        规模指数，例如：sh.000016 上证50，sh.000300 沪深300，sh.000905 中证500，sz.399001 深证成指等；
        注意指数没有分钟线数据... ...怎么办？
        ie. 'sh.000023' goes to 'sh.000001' # 上证综指 
            'sz.300333' goes to 'sz.399106' # 深圳综指
            'hk.00700' goes to 'HSI'        # 恒生指数
            'us.######' goes to 'NASDAQ' or 'DJX' 
        '''
        market = ticker.split(".")[0]
        # match market:   # requires python 3.10 or higher version
        #     case 'sh': 'sh.000001'
        #     case 'sz':  'sz.399106'
        #     case 'hk':  'HSI'
        #     case 'us':  'DJX'
        if market == 'sh': mkt = 'sh.000001'
        elif market == 'sz': mkt = 'sz.399106'
        elif market == 'hk': mkt = 'HSI'
        else: print("invalid market label")

        return mkt 

    def etf(self, ticker) -> str:
        '''股票所属etf基金'''
        return ''

class BaostockDataWorker(DataWorker):
    def __init__(self) -> None:
        super().__init__()
        self.login = bs.login(user_id="anonymous", password="123456")
        self.ds = DataStorage()
        self.calendar = self.market_calendar()
        self.calendar = self.calendar[self.calendar.is_trading_day == "1"]  # 只留下交易日

    # @property
    def all_tickers(self) -> pd.DataFrame :
        tickers = bs.query_stock_basic().get_data()  # 获取最新证券基本资料，可以通过参数设置获取对应证券代码、证券名称的数据
        self.ds.save_tickers(tickers)
        return tickers

    def minute(self,ticker,start_date = None, end_date = None, ktype='5') -> pd.DataFrame :
        ''' download 5-minutely data if ticker available
        http://baostock.com/baostock/index.php/Python_API文档
        '''
        start_date = self.begin if start_date is None else start_date
        end_date = datetime.today().strftime("%Y-%m-%d") if end_date is None else end_date
        if ktype in ['5','10','15','30','60']:
            rs = bs.query_history_k_data_plus(ticker, "date,time,code,open,high,low,close,volume,amount,adjustflag",
                start_date = start_date, end_date = end_date, frequency = ktype, adjustflag="2")
        else:  # ktype in ['d','w','m']
            rs = bs.query_history_k_data_plus(ticker, "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,pctChg",
                start_date = start_date, end_date = end_date, frequency = ktype, adjustflag="2")

        if rs.error_code == '0':
            self.stock = rs.get_data() 
        else:
            raise Exception("something wrong with get_data():" + rs.error_msg)
        
        # 发现时间太长的，volume和amount有空子串，导致转换出错，替换成"0"
        self.stock.replace(to_replace=r"^\s*$", value ="0", regex=True, inplace=True)
        self.stock[OCLHVA] = self.stock[OCLHVA].astype('float64')  # Baostock给出的是object，不是float的要转成float
        self.stock.rename(columns = BAOSTOCK_MAPPING, inplace = True) 
        return self.stock
    
    def market_calendar(self, begin = None):
        '''return market trading days'''
        begin = begin if begin else self.begin
        assert pd.to_datetime(begin), "wrong format of `begin`, must be `%Y-%m-%d`"
        rs = bs.query_trade_dates(start_date = begin, end_date = datetime.today().strftime("%Y-%m-%d"))
        return  rs.get_data()
    
    def latest(self,ticker, ktype = '5', days = WINDOW_SIZE):
        trade_days = self.calendar.tail(days)
        start_date, end_date = trade_days.calendar_date.min(), trade_days.calendar_date.max()
        return self.minute(ticker, start_date, end_date, ktype)

    def minute_mkt(self,ticker,ktype='5') -> pd.DataFrame:
        ''' downloads 5-min with market data if ticker available
        常用的合并方法包括append、assign、combine、update、concat、merge、join等
        '''
        rs1 = self.minute(ticker, ktype)
        mkt = self.market(ticker)
        begin = rs1.date.min
        end = rs1.date.max
        rs2 = self.minute(mkt, begin, end, ktype = 'd') # only ktype = 'd' is available for indices in Baostock
        rs2.rename(columns=dict(zip(OCLHVA, MKT_OCLHVA)),inplace = True) 
        rs2.rename(columns={'code':"mkt_code"},inplace = True) 
        return rs1, rs2 # returns rs1, rs2 seperately
        # return rs1.merge(rs2,how='left', left_on=["date","time"],right_on=["date","time"])  # 按列合并
    
    def actual_days(self, df:pd.DataFrame):
        ''' filter actual trade days for `df`.
        df : the dataframe to retrieve min,max date
        requires `df` has a string or datetime column `date`
        '''
        assert 'date' in df.columns, "`df` must have a string or datetime column `date`"
        n, x = df.date.min(), df.date.max()
        trade_days = self.calendar       
        date_filter = (n <= trade_days.calendar_date) & (trade_days.calendar_date <= x)
        return trade_days[date_filter]   # 1. 只剩交易日 2. 范围只剩与本股票相关的

    def save(self, rs:pd.DataFrame) -> bool:
        return self.ds.save_raw(rs)

class TushareDataWorker(DataWorker):
    def __init__(self) -> None:
        super().__init__()
        ts.set_token(TS_TOKEN)
        self.pro = ts.pro_api()
        self.ds = DataStorage()
        # ds.empty_raw()    # 清空raw数据库

    @property
    def all_tickers(self):
        return self.pro.stock_basic(exchange='', list_status='L', fields='ts_code, name, area,industry,list_date')

    def get(self,ticker ,asset='I',freq='1min',start_date='2020-01-07 09:00:00',end_date='2020-04-01 15:30:00'):
        ticker = ticker if ticker else self.all_tickers.sample(1).iloc[0].ts_code # retrieved was a set of records, needs a iloc[0] to take out the exact one.
        return self.pro.stk_mins(ts_code=ticker,freq=freq,start_date=start_date, end_date=end_date)
    
    def batch_get(self,ts_code)-> pd.DataFrame:
        end = datetime.now()
        for i in count():
            start = end - timedelta(days=60)    # 按照这个[start, end]获取的数据本次的最后一条与下一次的第一条是重复的。最好是[start,end)
            try:
                df = self.get(ts_code=ts_code, start_date = start.strftime("%Y-%m-%d %H:%M:%S"),end_date= end.strftime("%Y-%m-%d %H:%M:%S"))
                dw.save_raw(df[1:],if_exists='append') # 去重；也可以放在数据库层去重，还可以在train前提取数据时去重，甚至...可以不去重？
                x,y = df.iloc[0].trade_time, df.iloc[-1].trade_time
                print("%d\t(%s <- %s]\t%d records saved."%(i,x,y,len(df)))  # [y,x) 前闭后开
            except Exception as e:
                print(e)

            end = datetime.strptime(y,"%Y-%m-%d %H:%M:%S")
            if start < datetime(2009,1,1): 
                break
    

if __name__ == "__main__":
    DATABASE_PATH = os.getcwd()     # 解决方法
    print("----database:"+DATABASE_PATH +"/"+ DATABASE + "----")
    dw = BaostockDataWorker()
    t = dw.all_tickers()
    t[['type','status']] = t[['type','status']].astype("int16")
    # t[['ipoDate','outDate']] = pd.to_datetime(t[['ipoDate','outDate']])
    t[['code','code_name']] = t[['code','code_name']].astype("string")
    t0 = t[(t.type == 1)&(t.status ==1)].sample(1).iloc[0].code
    rs = dw.minute(t0)
    dw.save(rs)
    print(rs.columns)
    print(rs.describe())
    print(rs)
    # import dtale
    # dtale.show(rs).open_browser()