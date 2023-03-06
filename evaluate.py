from this import d
import pandas as pd
import os


class Evaluator():
    '''
        dataworker获取raw的部分；preprocessor添加了一部分如landmark和若干技术指标用于embedding；trainer添加了['action','reward']
        需要评估的内容有：
        - reward 在一个股票学习中是不是呈总体上升趋势。在历史以来的学习中，reward有无上升趋势？这可能需要另外有个表保存 
        - action 与 landmark的重合度如何？ optional
        - 假设有100%资产，按照action操作，资产变化情况如何？增加一个字段d.asset_pctChg
        注：reward的设计，已经充分体现了动作的精度：越接近同买同卖点分数越高，因此不再需要分类指标来表达
    '''

    def __init__(self, df, predicted=None, episode=None ):
        self.df = df
        self.evaluated = pd.DataFrame(columns = ['ticker','tp','fp','tn','fn','accuracy','precision','recall','f1_score','tpr','fpr','annual_return'])

    def asset_change(self):
        ''' calculate asset change by percent at each action
            'short operation' means to operate reversely (borrow shares to sell at higher price then buy to return at lower)
        before: self.df.columns = ['ticker','date','close','action','reward']
        after: self.df.columns = ['ticker','date','close','action','reward','real_action','change_wo_short', 'change_w_short','asset_wo_short','asset_w_short']
        4 columns added.
        '''
        # def reduce_same(d1):  # action buy和sell的action应该交替出现，所以应该合并连续的sell或连续的buy
        #     i = 0
        #     p1 = d1.iloc[i]
        #     d2 = pd.DataFrame()
        #     d2 = d2.append(p1)
        #     while i< len(d1)-1:
        #         p2 = d1.iloc[i+1]
        #         if p1.action != p2.action:
        #             d2 = d2.append(p2)  # 如果action不同，则收集起来，并将p1移到这个位置
        #             p1 = p2
        #         i += 1  # 如果action相同，则跳过
            
        #     return d2
        # # 忽然想到更好的做法替代reduce_same(d1)
        # # d1[['next_date','next_action']] = d1[['date','action']].shift(1)
        # # d1['real_action'] = d1.apply(lambda r: r.action if r.action != r.next_action else 0, axis=1)
        # # 以上代码待测试

        # actioned = self.df[self.df.action.isin([1,-1])]
        # actioned = reduce_same(actioned)
        # self.df["real_action"] = self.df.apply(lambda x: x.action if x.name in actioned.index else 0, axis = 1)
        # # real_action基本用于visualize中的绘制灰底，也许也可以完全用action搞定
        
        # def func(x,y):  
        #     '''add columns `without_short` and `with_short`'''
        #     # 没有做空的话，卖出后股票的变化率是1， .iloc[x].real_action == 1 先买后卖，== -1 是先卖后买
        #     self.df.loc[x:y, 'change_wo_short'] = \
        #         self.df.close/self.df.close.shift(1) if self.df.iloc[x].real_action == 1 else 1    
        #     # 有做空的话，卖出后股票变化率跟原来相反
        #     self.df.loc[x:y, 'change_w_short'] = \
        #         self.df.close/self.df.close.shift(1) if self.df.iloc[x].real_action == 1 else self.df.close.shift(1)/self.df.close
        
        # d1, d2 = actioned.iloc[:-1].index, actioned.iloc[1:].index  # 获取索引并配对拼接
        # # actioned['next_day'] = actioned.date.shift(-1)   # 这样不是更简单吗？
        # # actioned.apply(lambda a:(a.date, a.next_day) if a.real_action==-1 else None, axis = 1)
        # self.df["change_wo_short"] = self.df["change_w_short"] = 1   

        # [func(x,y) for (x,y) in zip(d1,d2)]

        # self.df[['change_wo_short','change_w_short']] = self.df[['change_wo_short','change_w_short']].fillna(1)

        d = self.df
        d['change_wo_short'] = d['change_w_short'] = d.close/d.close.shift(1) # 只做多情况下与上一天的变化比例，会在第一行留下nan
        d.loc[d.action == -1,'change_wo_short'] = 1     # 不做空的日子，与上一天相比没变化
        d.loc[d.action == -1,'change_w_short'] **= -1   # 有做空的日子，与上一天的比率取倒数
        d[['change_wo_short','change_w_short']] = d[['change_wo_short','change_w_short']].fillna(1)# 补齐第一行的Nan
        self.df = d
        return self
    
    def annual_retuan(df:pd.DataFrame):
        '''requires: df has columns ['date','close','change_wo_short','change_w_short']
        calculate annual return with `change_wo_short` and `change_w_short`, compared with `close`
        '''

        pass

    def class_perf(self):
        '''performance as classification'''
        # 作为二分类问题，这里应该用action而非real_action
        # 例如fn的本质应该是漏报，即action not in [-1,1]的时候，landmark in [-1,1]
        tp = self.df[(self.df.action==1)&(self.df.reward>0)].shape[0]   # 动作为买入，而判断正确
        fp = self.df[(self.df.action==1)&(self.df.reward<0)].shape[0]   # 动作为买入，但判断错误
        tn = self.df[(self.df.action==-1)&(self.df.reward>0)].shape[0]  # 动作为卖出，而判断正确
        fn = self.df[(self.df.action==-1)&(self.df.reward<0)].shape[0]  # 动作为卖出，但判断错误
        accuracy = (tp+tn)/(tp+tn+fp+fn)
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        f1_score = 2*precision*recall/(precision+recall)
        tpr = recall
        fpr = fp/(fp+tn)
        self.evaluated.loc[len(self.evaluated)] = [self.df.iloc[0].ticker,tp,fp,tn,fn,accuracy, precision, recall, f1_score, tpr, fpr,0]
        return self.evaluated
    
    def regr_perf(self):
        ''' performance as regression
            wait until df has a column `reward_predicted`, generated by a non-linear regression model
        '''
        
        pass

from globals import test_result, summary
from pathlib import Path
import sys

def run(obj):
    data_folder = Path(f"{test_result}/{obj}")
    files = os.listdir(str(data_folder))  # 目录下所有文件,
    files = [f for f in files if os.path.splitext(f)[1] == '.csv']  # 只选择 .csv 文件,
    files.remove(summary) if summary in files else files
    print(f"Evaluating strategy {obj} with {files}")
    df_list = pd.DataFrame()
    for f in files: 
        file_to_open = data_folder / f
        df = pd.read_csv(file_to_open, index_col=0)
        ev = Evaluator(df)
        ev.asset_change().df.to_csv(file_to_open)  # 保存asset_change()的结果到原f 
        performance = ev.class_perf()   # 返回class_perf()的结果给performance
        df_list = df_list.append(performance)   # 

    df_list.to_csv(data_folder/f'{summary}')

if __name__ == "__main__":
    # obj = 'Chandelier'  # 改成你自己的策略
    if len(sys.argv)>=2:
        obj = sys.argv[1]
        run(obj)    
    else:
        dirs = os.listdir(f"{test_result}")  # Test目录下的子目录
        dirs = [run(d) for d in dirs if not os.path.isfile(d)]

