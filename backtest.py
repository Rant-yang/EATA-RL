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
from datetime import datetime
import pandas as pd
from tqdm import trange
import os
from globals import WINDOW_SIZE
from globals import test_result

def run(agent, row, days) -> pd.DataFrame:
    env = StockmarketEnv(row, days)
    s = env.reset()
    result = pd.DataFrame(columns=['ticker','date','close','action','reward'])
    for _ in trange(len(env.trade_days) - WINDOW_SIZE ): # for each row of a stock data, days copied from data days
        a = agent.choose_action(s)  # a class method of class Bandwagon
        s_, r, done, info = env.step(a)
        result.loc[len(result)] = [*info.values(), a, r] # append new row for df `result`.
        # agent.memory.add(s,a,r)
        # agent.learn() # 
        s = s_
        if done:
            break

    return result

# "./" will be added in front of each directory
def check_and_make_directories(directories: list[str]):
    for directory in directories:
        if not os.path.exists("./" + directory):
            os.makedirs("./" + directory)
#%%
from globals import test_result
from pathlib import Path

def test(Obj:object, stock_list:pd.DataFrame):
    """
    Test function for stock objects.

    This function iterates over a DataFrame of stocks, processes each stock using the provided object,
    and saves the result to a CSV file. It also prints the progress and performs an evaluation at the end.

    Args:
    Obj (object): The object to be used for processing each stock.
    stock_list (pd.DataFrame): A DataFrame containing the list of stocks to be processed.
        # columns = ['code', 'name','weight', 'sector'] 分别代表：股票代码，股票名称，权重，所属板块的指数代码
    """    
    obj = Obj(stock_list)   
    df = obj.stock_list
    data_folder = Path(f"{test_result}/{obj.__name__}")
    today = datetime.today().strftime("%Y%m%d.%H%M")   # make sure it does not go beyond to the next day
    print(f"*** Testing {obj.__name__} ***")
    for i, row in df.iterrows(): # for each SZ50 constituent
        print(f"#{i} Processing {row.code} {row['name']}")
        result = run(obj, row, days= 2000)   
        check_and_make_directories([str(data_folder)])
        file_to_open = data_folder / f"{today}{row.code}.csv"
        result.to_csv(file_to_open)
        # or we do evaluation here
        # from evaluate import Evaluator
        # ev = Evaluator(df)
        # ev.asset_change().df.to_csv(file_to_open)  # 保存asset_change()的结果到原f 

    print(f"Test done. Check the folder {obj.__name__}")
    os.system(f"python evaluate.py {obj.__name__}")
    print("Evaluation done. Check the visualizer")

def inference(Obj:object, stock_list:pd.DataFrame):
    obj = Obj(stock_list)
    score = obj.vote()
    print(f"score = {score}, Buy(1) or Sell(-1)?", obj.etf_action(score))

if __name__ == "__main__":
    # df = pd.read_excel("000016(full).xls", dtype={'code':'str'}, header = 0)
    # df = df.sample(2) # 开发时用，测试agent的时候将此行注释即可
    # test(Obj, df)
    pass
