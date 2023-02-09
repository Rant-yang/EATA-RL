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

    def asset_change(self) -> pd.DataFrame:
        ''' calculate asset change by percent at each action
            'short operation' means to operate reversely (borrow shares to sell at higher price then buy to return at lower)
        before: self.df.columns = ['ticker','date','close','action','reward']
        after: self.df.columns = ['ticker','date','close','action','reward','change_wo_short', 'change_w_short','asset_wo_short','asset_w_short']
        4 columns added.
        '''
        def reduce_same(d1):  # action buy和sell的action应该交替出现，所以应该合并连续的sell或连续的buy
            i = 0
            p1 = d1.iloc[i]
            d2 = pd.DataFrame()
            d2 = d2.append(p1)
            while i< len(d1)-1:
                p2 = d1.iloc[i+1]
                if p1.action != p2.action:
                    d2 = d2.append(p2)  # 如果action不同，则收集起来，并将p1移到这个位置
                    p1 = p2
                i += 1  # 如果action相同，则跳过
            
            return d2

        actioned = self.df[self.df.action.isin([1,-1])]
        actioned = reduce_same(actioned)

        d1, d2 = actioned.iloc[:-1].index, actioned.iloc[1:].index  # 获取索引并配对拼接
        df["change_wo_short"] = 1  
        df["change_w_short"] = 1   

        def func(x,y):  
            '''add columns `without_short` and `with_short`'''
            # 没有做空的话，卖出后股票的变化率是1
            self.df.loc[x:y, 'change_wo_short'] = \
                self.df.close/self.df.close.shift(1) if self.df.iloc[x].action == 1 else 1
            # 有做空的话，卖出后股票变化率跟原来相反
            self.df.loc[x:y, 'change_w_short'] = \
                self.df.close/self.df.close.shift(1) if self.df.iloc[x].action == 1 else self.df.close.shift(1)/self.df.close

        [func(x,y) for (x,y) in zip(d1,d2)]

        # df.change1.plot(kind="hist")
        self.df['asset_wo_short'] = self.df.change_wo_short.cumprod()  # 不做空情况下资产变化曲线
        self.df['asset_w_short'] = self.df.change_w_short.cumprod()  # 不做空情况下资产变化曲线
        return self

    def class_perf(self):
        '''performance as classification'''
        # action为根据策略的动作反应，暗含predict的意思，相当于预测值；reward事后诸葛亮，根据波峰波谷得出原本应该的动作，相当于真实值
        # 设action中，买入为正类预测p，卖出为负类预测n
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
        self.evaluated[['ticker']] = self.df.iloc[0].ticker
        self.evaluated[['tp','fp','tn','fn']] = tp,fp,tn,fn
        self.evaluated[['accuracy','precision','recall','f1_score','tpr','fpr']] = accuracy, precision, recall, f1_score, tpr, fpr
        self.evaluated[['annual_return']] = 0

        return self.evaluated


if __name__ == "__main__":
    files = os.listdir()  # 目录下所有文件,
    files = [f for f in files if os.path.splitext(f)[1] == '.csv']  # 只选择 .csv 文件,
    print(f"Examing experiments on {files}")
    df_list = pd.DataFrame()
    for f in files: 
        df = pd.read_csv(f, index_col=0)
        ev = Evaluator(df)
        ev.asset_change().df.to_csv(f)  # 保存asset_change()的结果到原f 
        performance = ev.class_perf()   # 返回class_perf()的结果给performance
        df_list = df_list.append(performance)   # 

    df_list.to_csv('evaluated.csv')