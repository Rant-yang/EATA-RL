import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from globals import summary, test_result

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
        self.obj = self.agents[2]    # 这行开始改成循环
        files = os.listdir(f'{test_result}/{self.obj}')  # 目录下所有文件
        files = [f for f in files if os.path.splitext(f)[1] == '.csv']  # 只选择 .csv 文件,
        files.remove(summary) if summary in files else files
        print(f"Testing strategy {self.obj} with {files}")
        self.perf = pd.read_csv(f'{test_result}/{self.obj}/{summary}', index_col= 0)
        self.dfs = [pd.read_csv(f'{test_result}/{self.obj}/{f}', index_col=0) for f in files]
        # 下一步改为根据agents，遍历指定目录

    def process(self, df):
        df = df.tail(200)   #选最后200交易日的数据，避免最后做出的图过于拥挤。也可以将天数作为初始参数之一
        self.df = df
        self.data_all = df.shape[0]
        # self.df2 = self.df.set_index('date')   #使用streamlit接口画图需要以date作为索引
        self.ticker = df.ticker.iloc[0]
        self.record = str(df.shape[0])  # 一共几天交易日的数据

        # 乘上close，使两天资产线和股票的收盘价线同一起点
        df['change_wo_short'].iloc[0] = 1  # 第一行赋值为1，以这个为起点，后面是相对于上一天比率
        df['change_w_short'].iloc[0] = 1    
        self.asset_wo_short = df.close.iloc[0] * df.change_wo_short.cumprod()
        self.asset_w_short = df.close.iloc[0] * df.change_w_short.cumprod()


        # 计算最后一天的资产
        self.asset_wo = df.close.iloc[0]* self.asset_wo_short.iloc[-1]      #不做空 最新一日资产
        # self.chg_wo = round(df.change_wo_short.iloc[-1],2)
        self.asset_w = df.close.iloc[0]* self.asset_w_short.iloc[-1]     #做空 最新一日资产
        # self.chg_w = round(df.change_w_short.iloc[-1],2)

        # 分别挑出action为买或卖的收盘价，以便于画图标注
        test = df.real_action * df.close
        self.buy_actions = test.apply(lambda x: x if x>0 else None)     # sell的位置为None，维持序列长度不变
        self.sell_actions = test.apply(lambda x: -x if x<0 else None)   # buy的位置为None，维持序列长度不变
        # self.buy_actions = [t for t in test if t>0]
        # self.sell_actions = [-t for t in test if t<0]

        self.tick_spacing = 10 #设置横坐标日期的间隙，避免重叠


    def run(self):
        st.title(f"Testing {self.obj}")
        st.header("Summary")
        st.dataframe(self.perf)
        [st.sidebar.text(a) for a in self.agents]

        for df in self.dfs:
            self.process(df)
            # st.subheader("CLOSE & ASSET GRAPH")
            st.metric(label="Ticker", value = self.ticker, delta = 'latest '+ self.record +' days')

            # 年化利率计算
            col1, col2 = st.columns(2)
            col1.metric(label="Annual return - Short",
                        value = round(self.asset_w / self.data_all * 250, 2)) #一年的交易日250天
            col2.metric(label="Annual return - No Short",
                        value = round(self.asset_wo / self.data_all * 250, 2))

            # 画图 close+asset+action：左边close，右边asset
            fig = plt.figure(figsize=(15,8))
            ax1 = fig.add_subplot(111)
            ax1.plot(self.df.date, self.asset_wo_short, 'm-.', label="without short")
            ax1.plot(self.df.date, self.asset_w_short, 'g-', label="with short")
            ax1.legend(loc=1)
            ax1.set_ylabel('Assets change/Close')

            ax2 = ax1 #.twinx()
            ax2.plot(self.df.date, self.df.close, 'b', label = "Price", alpha=0.1)
            ax2.fill_between(self.df.date, self.df.close.min(), self.df.close, color = 'b', alpha = 0.1)
            ax2.scatter(self.df.date, self.buy_actions, label='buy', color='red', marker="^")
            ax2.scatter(self.df.date, self.sell_actions, label='sell', color='green', marker ="v")
            ax2.legend(loc=2)
            ax2.set_xticklabels(labels=self.df.date, rotation=90) #不知道为旋转的变化显示不出来
            ax2.xaxis.set_major_locator(ticker.MultipleLocator(self.tick_spacing))
            # ax2.set_ylabel('Close')
            ax2.set_xlabel('Date')
            st.pyplot(fig)

if __name__=='__main__':
    WebServer().run()