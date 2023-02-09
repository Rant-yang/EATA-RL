''' following a RL paradigm, where an agent interacts with a env by .action() and .step()
1. `collect` 50 stocks(sz50 constituent) or 300 stocks(hs300 constituent) in the market, 
  and `label` each stock's sector(index code). note: bs.query_stock_basics() doesn't have the sector index code.
  we have to do it by hand.
2. `construct` a RL environment, especially .step(), .reward() methods
3. `predict` iteratively with bandwagon.choose_action()
3. `save` the predicted result as .csv, columns =  ['ticker', 'date', 'close', 'score',  'action', 'pct_chg']
4. `evaluate` the predicted result, calculating the asset_change, reward (in evaluate.py)
5. `visualize` the evaluated result (in visualize.py)
'''
from env import StockmarketEnv
from bandwagon import Bandwagon as agent     # 换其他模型，只需要修改这句。例如 from chandelier import ChandelierExit as agent
from datetime import datetime
import pandas as pd
from tqdm import trange

def run(row, days) -> pd.DataFrame:
    env = StockmarketEnv(row, days)
    s = env.reset()
    result = pd.DataFrame(columns=['ticker','date','close','action','reward'])
    for _ in trange(days): # for each row of a stock data, days copied from data days
        a = agent.choose_action(s)  # a class method of class Bandwagon
        s_, r, done, info = env.step(a)
        result.loc[len(result)] = [*info.values(), a, r] # append new row for df `result`.
        s = s_
        if done:
            break

    return result
    
#%%
import importlib
import sys
if __name__ == "__main__":
    # agent = importlib.import_module(sys.argv[2])
    print(f"Testing {agent.__name__}")
    today = datetime.today().strftime("%Y%m%d.%H%M")   # make sure it does not go beyond to the next day
    df = pd.read_excel("000016(full).xls", dtype={'code':'str'}, header = 0)
    df = df.sample(2) # 开发时用，测试agent的时候将此行注释即可
    for i, row in df.iterrows(): # for each SZ50 constituent
        print(f"#{i} processing {row.code} {row['name']}")
        result = run(row, days= 1000)   
        result.to_csv(f"{agent.__name__}/{today}{row.code}.csv")
