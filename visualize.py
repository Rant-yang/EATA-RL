import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from globals import summary, test_result
from pathlib import Path
from functools import partial

def load_css(file_name:str = "streamlit.css")->None:
    """
    Function to load and render a local stylesheet
    """
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


class WebServer:
    def __init__(self):
        dirs = os.listdir(f"{test_result}")  # Test目录下的子目录
        self.agents = [d for d in dirs if not os.path.isfile(d)]
        self.obj = self.agents[0]    
        self.get_folder()

    def get_folder(self):
        data_folder = Path(f"{test_result}/{self.obj}")
        files = os.listdir(data_folder)  # 目录下所有文件
        files = [f for f in files if os.path.splitext(f)[1] == '.csv']  # 只选择 .csv 文件,
        files.remove(summary) if summary in files else None
        print(f"\nVisualizing strategy {self.obj} with {files}")
        self.perf = pd.read_csv(data_folder/f'{summary}', index_col= 0) # 需要加个文件不存在的判断，self.perf = None
        self.dfs = [pd.read_csv(data_folder/f'{f}', index_col=0) for f in files]
        # 下一步改为根据agents，遍历指定目录

    def process(self, df, days = 240):
        d = df.tail(days).copy()   #选最后n个交易日的数据(大致1年），避免最后做出的图过于拥挤。
        self.data_all = d.shape[0]
        self.ticker = d.ticker.iloc[0]
        self.record = str(d.shape[0])  # 一共几天交易日的数据

        # 乘上close，使两天资产线和股票的收盘价线同一起点
        d[0,'change_wo_short'] = 1  # 第一行赋值为1，以这个为起点，后面是相对于上一天比率
        d[0,'change_w_short'] = 1   # 第一行赋值为1，以这个为起点，后面是相对于上一天比率 
        self.asset_wo_short = d.close.iloc[0] * d.change_wo_short.cumprod()
        self.asset_w_short = d.close.iloc[0] * d.change_w_short.cumprod()

        # calculate annual return here 
        self.asset_wo = self.asset_wo_short.iloc[-1]      #不做空 最新一日资产
        self.asset_w =  self.asset_w_short.iloc[-1]     #做空 最新一日资产

        # 分别挑出action为买或卖，目的是为了做区间绘图
        d['last_action'] = d['action'].shift(1) # (action, 昨天的action)
        self.action_days = d[d.action != d.last_action][['date','close','action','reward']] 
        # self.action_days['date'] = pd.to_datetime(self.action_days.date)    # 改成datetime，才能和self.df.date比较
        self.action_days['next_day'] = self.action_days.date.shift(-1)      # (.date, .next_day) 确定了灰底的范围
        self.action_days['next_day'].fillna(d.iloc[-1].date, inplace = True) # 用d最后一行的date补齐最后一行的空值

        self.df = d 
        return d
    
    def attach_grey(self, sdf, axes):
        '''
            requires: sdf[['action','norm_close']] and a `date` as index
        '''
        sdf['last_action'] = sdf.action.shift(1)    # compared with yesterday
        action_days = sdf[sdf.action != sdf.last_action] # find the flex
        action_days = action_days.reset_index() # 将index恢复为column
        action_days['next_day'] = action_days.shift(-1).date
        action_days['next_day'] = action_days['next_day'].fillna(sdf.index[-1]) # 用d最后一行的date补齐最后一行的空值
        short_days = action_days[action_days.action == -1]['next_day']
        short_days = short_days.reset_index()   # 做空的期间 (data, next_day)

        def plot_range(x, ax, df, floor):
            # it requires x.action, x.date , x.next_day, dates, x.close
            ax.fill_between(df.index, floor, df.norm_close, (x.date < df.index) & (df.index< x.next_day), color = "k", alpha = .1)
        
        for ax in axes:
            pr = partial(plot_range, df = sdf, floor=sdf.norm_close.min(), ax = ax)     # 生成一个偏函数，供apply使用
            short_days.apply(pr, axis=1)
        
        return short_days

    def run(self):
        self.obj = st.sidebar.radio('Choose one',self.agents)
        self.get_folder()
        st.markdown(f'''<style> .appview-container .main .block-container{{
            padding-top: {1}rem; 
            padding-right: {0}rem; 
            padding-left: {0}rem; 
            padding-bottom: {1}rem;
            }}</style>''', unsafe_allow_html=True)

        st.title(f"Performance of {self.obj}")

        if self.perf is not None:
            st.header("Summary")

            # histograms of metrics in a 2*3 grid
            my_list = ['accuracy','precision','recall', 'f1_score', 'fpr','reward']
            p = self.perf[my_list]
            # f = plt.figure(figsize=(10,5))
            f, axes = plt.subplots(1,1,figsize=(10,5))
            p.plot(ax = axes, subplots = True, kind = 'hist', bins=50, layout = (2, 3), \
                legend = False, title = p.columns.to_list(), colormap='viridis', alpha = 1, sharex=True)

            # f,axes = plt.subplots(nrows=2,ncols=3,figsize=(15,8))
            # from itertools import count
            # for i,m in zip(count(start = 0, step = 1), my_list):
            #     f.axes[i].hist(self.perf[m], bins = 20, alpha = 0.8) 
            #     f.axes[i].set_title(m)
            st.pyplot(f)
            st.dataframe(self.perf)
        
        # 对单个股票做图
        for df in self.dfs:
            d = self.process(df)    # 用d，好过用self.df
            # st.subheader("CLOSE & ASSET GRAPH")
            st.metric(label="Ticker", value = self.ticker, delta = 'latest '+ self.record +' days')
            
            # 年化利率计算
            col1, col2 = st.columns(2)
            col1.metric(label="Annual return - Short",
                        value = round(self.asset_w / self.data_all * 250, 2)) #一年的交易日250天
            col2.metric(label="Annual return - No Short",
                        value = round(self.asset_wo / self.data_all * 250, 2))
            
            # 确定绘图的地板和天花板
            floor = min(self.df.close.min(), self.asset_w_short.min(), self.asset_wo_short.min())
            ceiling = max(self.df.close.max(), self.asset_w_short.max(), self.asset_wo_short.max())

            # self.df['date'] = pd.to_datetime(self.df.date)  # 不转成datetime类型x轴会很拥挤
            # 画图 close+asset+action
            fig = plt.figure(figsize=(15,8))
            ax1 = fig.add_subplot(111)
            ax1.plot(self.df.date, self.asset_wo_short, 'm-.', label="without short")
            ax1.plot(self.df.date, self.asset_w_short, 'g-', label="with short")
            ax1.legend(loc=1)
            ax1.set_ylabel('Assets change/Close')
            ax2 = ax1 #.twinx()
            ax2.plot(self.df.date, self.df.close, 'b', label = "Price", alpha=0.1)          # close
            ax2.fill_between(self.df.date, floor, self.df.close, color = 'b', alpha = 0.1)  # close 面积填充

            # 采用灰底来表示做空时间，白底做多
            # self.attach_grey(df,[ax1])
            def plot_range(x):
                if x.action == -1:  # 区间起点为卖出
                    ax2.fill_between(self.df.date, floor, ceiling, (x.date <self.df.date) & (self.df.date< x.next_day), color = "k", alpha = 0.1)
                
                ax2.annotate(x.date, xy = (x.date, ceiling), rotation = -45, fontsize = 8, arrowprops=dict(arrowstyle="->",connectionstyle="arc3,rad=.2"))
                ax2.annotate('%.3f'%x.reward, xy = (x.date, x.close), rotation = -30, fontsize = 8, arrowprops=dict(arrowstyle="->",connectionstyle="arc3,rad=.2"))
                # ax2.text(self.df.date, ceiling, self.df.date)
            self.action_days.apply(plot_range, axis = 1)
            # ---------
            ax2.legend(loc=2)
            # ax2.set_ylabel('Close')
            ax2.set_xlabel('Date')
            st.pyplot(fig)
            plt.close()

if __name__=='__main__':
    WebServer().run()