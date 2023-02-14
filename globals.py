# CAUTION: globals variables could be imported by other modules widely, so don't do it wildly. 
# Any modification may lead to unpredictable consequence!

import os
MAIN_PATH = os.path.dirname(__file__)  # 当前文件所在的目录
os.chdir(MAIN_PATH)
print(f"Working in {MAIN_PATH}")

#%%
from collections import namedtuple
Action = namedtuple('Action', ['buy', 'hold', 'sell'])
action = Action(1,0,-1)     # however, gym.spaces.Discrete(3) allows [0,1,2] only.
Flex = namedtuple('Flex', ['peak', 'waist', 'bottom'])
flex = Flex(-1,0,1)
#%% 
WINDOW_SIZE = 20

#%% prefix
DATE_FORMAT = "%Y-%m-%d"
MARKET_PREFIX = "mkt_"
ETF_PREFIX = "etf_"
SECTOR_PREFIX = "sct_"
#%% column definitions
TD = ['ticker', 'date']
TDT = ['ticker', 'date','time']
OCLHVA = ['open',  'high', 'low', 'close', 'volume', 'amount']
Normed_OCLHVA = [x+"_" for x in OCLHVA]     # normalized ohlcva
MKT_OCLHVA = OCLHVA 
mkt_oclhva_normed = [MARKET_PREFIX+x for x in Normed_OCLHVA]

REWARD = ['buy_reward', 'hold_reward', 'sell_reward']
WEEKDAY = ['dayofweek']

# indicators collection
# indicators = ['kdjk', 'kdjd', 'kdjj', "rsi_6", "rsi_12", "rsi_24",'cr',"boll","boll_ub","boll_lb","wr_10","wr_6","cci","dma"] # 14
# indicators = ['kdjk', 'kdjd', 'kdjj', "rsi_6", "rsi_12", "rsi_24","macd","atr"] # https://github.com/jealous/stockstats
# "MFI","EMV","VR","PSY","OBV" are volume-concerned indicators
# indicators = ['close_5_ema', 'close_10_ema','kdj', "rsi_6","rsi_12","rsi_24","macds","macdh","atr","vr"] # https://github.com/jealous/stockstats
# indicators_after = ['kdjk', 'kdjd', 'kdjj', "rsi_6", "rsi_12", "rsi_24","macds","macdh","atr","vr"] # https://github.com/jealous/stockstats
indicators = ['close_5_ema', 'close_10_ema','rsi','atr'] # https://github.com/jealous/stockstats
mkt_indicators = [MARKET_PREFIX+x for x in indicators]
sct_indicators = [SECTOR_PREFIX+x for x in indicators]

#%%

import tushare as ts
TS_TOKEN = "72d1e47c3b0728a26bfc4a9f54132b195890fa843815f896708515f1"
ts.set_token(TS_TOKEN)

# data source mapping dictionary, used by pd.rename()
TUSHARE_MAPPING = {"ts_code":"ticker","trade_date":"date","vol":"volume"}
BAOSTOCK_MAPPING = {"code":"ticker"}

#%%
# # 上证50ETF成分股
# 上证50指数依据样本稳定性和动态跟踪相结合的原则，每半年调整一次成份股，调整时间与上证180指数一致。特殊情况时也可能对样本进行临时调整。
# 每次调整的比例一般情况不超过10％。样本调整设置缓冲区，排名在40名之前的新样本优先进入，排名在60名之前的老样本优先保留

SH50Index = "000016" # 上证50指数代码
SH50ETF = "510050" # 上证50ETF代码

summary = 'evaluated.csv'   # 统计文件名称
test_result = "Test"    # 测试结果存放目录