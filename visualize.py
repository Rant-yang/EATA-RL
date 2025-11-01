import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import empyrical as ep
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
        # KPI scope toggle
        kpi_scope = st.sidebar.radio('KPI scope', ['Window','Full'], index=0)
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
            # histograms of metrics (classification + empyrical metrics)
            base_cols = ['accuracy','precision','recall','f1_score','fpr','reward']
            extra_cols = ['ann_ret_wo','ann_ret_w','sharpe_wo','sharpe_w','max_dd_wo','max_dd_w']
            my_list = [c for c in base_cols + extra_cols if c in self.perf.columns]
            if len(my_list) > 0:
                p = self.perf[my_list]
                # dynamic layout: up to 3 cols per row
                import math
                n = len(my_list)
                ncols = min(3, n)
                nrows = math.ceil(n / ncols)
                f, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 3*nrows))
                # Normalize axes to flat iterable
                if isinstance(axes, plt.Axes):
                    axes = [axes]
                else:
                    axes = axes.flatten()
                for i, col in enumerate(my_list):
                    ax = axes[i]
                    p[col].plot(kind='hist', bins=50, ax=ax, title=col, color='tab:blue', alpha=0.8)
                    # Percent axis for annual return and drawdown columns
                    if col in ['ann_ret_wo','ann_ret_w','max_dd_wo','max_dd_w']:
                        ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
                    # Sharpe axis formatting
                    if col in ['sharpe_wo','sharpe_w']:
                        ax.xaxis.set_major_locator(ticker.MaxNLocator(6))
                        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
                # Hide unused axes
                for j in range(i+1, len(axes)):
                    axes[j].axis('off')
                st.pyplot(f)
                plt.close(f)
            # show table with percentage formatting for annual returns
            df_disp = self.perf.copy()
            percent_cols = [c for c in ['ann_ret_wo','ann_ret_w'] if c in df_disp.columns]
            if percent_cols:
                try:
                    st.dataframe(df_disp.style.format({c: '{:.2%}' for c in percent_cols}))
                except Exception:
                    # fallback to plain dataframe
                    st.dataframe(df_disp)
            else:
                st.dataframe(df_disp)
        
        # 对单个股票做图
        for df in self.dfs:
            d = self.process(df)    # 用d，好过用self.df
            # st.subheader("CLOSE & ASSET GRAPH")
            st.metric(label="Ticker", value = self.ticker, delta = 'latest '+ self.record +' days')
            
            # KPIs (Window-based using empyrical)
            # Build daily returns from change multipliers if available
            scope_df = self.df if kpi_scope == 'Window' else df
            r_wo = (scope_df['change_wo_short'].astype(float) - 1) if 'change_wo_short' in scope_df.columns else pd.Series(dtype=float)
            r_w = (scope_df['change_w_short'].astype(float) - 1) if 'change_w_short' in scope_df.columns else pd.Series(dtype=float)
            def safe_metric(func, r):
                try:
                    if r is None or r.empty or r.isna().all():
                        return float('nan')
                    r2 = r.replace([float('inf'), float('-inf')], 0).fillna(0)
                    return float(func(r2))
                except Exception:
                    return float('nan')

            ann_wo = safe_metric(ep.annual_return, r_wo)
            ann_w = safe_metric(ep.annual_return, r_w)
            shp_wo = safe_metric(ep.sharpe_ratio, r_wo)
            shp_w = safe_metric(ep.sharpe_ratio, r_w)
            mdd_wo = safe_metric(ep.max_drawdown, r_wo)
            mdd_w = safe_metric(ep.max_drawdown, r_w)

            st.subheader(f"KPIs ({kpi_scope})")
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric(label="AnnRet (No Short)", value = (f"{ann_wo*100:.2f}%" if pd.notna(ann_wo) else "-"))
            c2.metric(label="AnnRet (Short)", value = (f"{ann_w*100:.2f}%" if pd.notna(ann_w) else "-"))
            c3.metric(label="Sharpe (No Short)", value = (f"{shp_wo:.2f}" if pd.notna(shp_wo) else "-"))
            c4.metric(label="Sharpe (Short)", value = (f"{shp_w:.2f}" if pd.notna(shp_w) else "-"))
            c5.metric(label="MaxDD (No Short)", value = (f"{mdd_wo*100:.2f}%" if pd.notna(mdd_wo) else "-"))
            c6.metric(label="MaxDD (Short)", value = (f"{mdd_w*100:.2f}%" if pd.notna(mdd_w) else "-"))
            
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