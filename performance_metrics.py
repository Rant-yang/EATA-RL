"""
完整的交易评估指标
包括: AR, Sharpe, Sortino, MDD, WinRate, IRR, Volatility, Alpha, Beta
"""
import numpy as np
from typing import List, Dict
import pandas as pd


class TradingMetrics:
    """交易策略评估指标计算器"""
    
    def __init__(self, returns: np.ndarray, benchmark_returns: np.ndarray = None, risk_free_rate: float = 0.02):
        """
        参数:
            returns: 策略收益率序列 (daily returns)
            benchmark_returns: 基准收益率序列（用于计算Alpha/Beta）
            risk_free_rate: 无风险利率（年化）
        """
        self.returns = np.array(returns)
        self.benchmark_returns = np.array(benchmark_returns) if benchmark_returns is not None else None
        self.risk_free_rate = risk_free_rate
        self.trading_days_per_year = 252
    
    def annual_return(self) -> float:
        """年化收益率 (AR)"""
        total_return = (1 + self.returns).prod() - 1
        n_days = len(self.returns)
        n_years = n_days / self.trading_days_per_year
        
        if n_years == 0:
            return 0.0
        
        ar = (1 + total_return) ** (1 / n_years) - 1
        return ar
    
    def sharpe_ratio(self) -> float:
        """夏普比率
        Sharpe = (年化收益 - 无风险利率) / 年化波动率
        """
        if len(self.returns) == 0:
            return 0.0
        
        excess_returns = self.returns - self.risk_free_rate / self.trading_days_per_year
        
        if np.std(excess_returns) == 0:
            return 0.0
        
        sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(self.trading_days_per_year)
        return sharpe
    
    def sortino_ratio(self) -> float:
        """索提诺比率
        Sortino = (年化收益 - 无风险利率) / 下行波动率
        只考虑负收益的波动
        """
        if len(self.returns) == 0:
            return 0.0
        
        excess_returns = self.returns - self.risk_free_rate / self.trading_days_per_year
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return 0.0
        
        downside_std = np.std(downside_returns)
        sortino = np.mean(excess_returns) / downside_std * np.sqrt(self.trading_days_per_year)
        return sortino
    
    def max_drawdown(self) -> float:
        """最大回撤 (MDD)
        MDD = max((peak - valley) / peak)
        """
        if len(self.returns) == 0:
            return 0.0
        
        cumulative = (1 + self.returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (running_max - cumulative) / running_max
        
        mdd = np.max(drawdown)
        return mdd
    
    def calmar_ratio(self) -> float:
        """卡玛比率
        Calmar = 年化收益 / 最大回撤
        """
        mdd = self.max_drawdown()
        if mdd == 0:
            return 0.0
        
        ar = self.annual_return()
        return ar / mdd
    
    def volatility(self, annualized=True) -> float:
        """波动率
        
        参数:
            annualized: 是否年化
        """
        if len(self.returns) == 0:
            return 0.0
        
        vol = np.std(self.returns)
        
        if annualized:
            vol = vol * np.sqrt(self.trading_days_per_year)
        
        return vol
    
    def alpha_beta(self) -> tuple:
        """Alpha和Beta
        
        使用CAPM模型:
        R_p - R_f = α + β(R_m - R_f) + ε
        
        返回: (alpha, beta)
        """
        if self.benchmark_returns is None or len(self.benchmark_returns) != len(self.returns):
            return 0.0, 0.0
        
        # 超额收益
        excess_portfolio = self.returns - self.risk_free_rate / self.trading_days_per_year
        excess_market = self.benchmark_returns - self.risk_free_rate / self.trading_days_per_year
        
        # 计算Beta
        covariance = np.cov(excess_portfolio, excess_market)[0, 1]
        market_variance = np.var(excess_market)
        
        if market_variance == 0:
            return 0.0, 0.0
        
        beta = covariance / market_variance
        
        # 计算Alpha（年化）
        alpha = (np.mean(excess_portfolio) - beta * np.mean(excess_market)) * self.trading_days_per_year
        
        return alpha, beta
    
    def win_rate(self) -> float:
        """胜率
        胜率 = 正收益天数 / 总交易天数
        """
        if len(self.returns) == 0:
            return 0.0
        
        winning_days = np.sum(self.returns > 0)
        total_days = len(self.returns)
        
        return winning_days / total_days
    
    def profit_factor(self) -> float:
        """盈利因子
        PF = 总盈利 / 总亏损
        """
        profits = self.returns[self.returns > 0].sum()
        losses = abs(self.returns[self.returns < 0].sum())
        
        if losses == 0:
            return np.inf if profits > 0 else 0.0
        
        return profits / losses
    
    def irr(self, cash_flows: List[float] = None) -> float:
        """内部收益率 (IRR)
        
        如果不提供现金流，使用累计收益率近似
        """
        if cash_flows is not None:
            # 使用numpy的IRR计算
            try:
                import numpy_financial as npf
                return npf.irr(cash_flows)
            except ImportError:
                # 如果没有numpy_financial，用简化计算
                pass
        
        # 简化：使用年化收益率
        return self.annual_return()
    
    def get_all_metrics(self) -> Dict[str, float]:
        """计算所有指标"""
        alpha, beta = self.alpha_beta()
        
        metrics = {
            'Annual Return (AR)': self.annual_return(),
            'Sharpe Ratio': self.sharpe_ratio(),
            'Sortino Ratio': self.sortino_ratio(),
            'Max Drawdown (MDD)': self.max_drawdown(),
            'Calmar Ratio': self.calmar_ratio(),
            'Win Rate': self.win_rate(),
            'Volatility (Annual)': self.volatility(annualized=True),
            'Alpha': alpha,
            'Beta': beta,
            'IRR': self.irr(),
            'Profit Factor': self.profit_factor()
        }
        
        return metrics
    
    def print_metrics(self, title="Strategy Metrics"):
        """打印所有指标"""
        print("\n" + "="*60)
        print(f"{title}")
        print("="*60)
        
        metrics = self.get_all_metrics()
        
        for name, value in metrics.items():
            if 'Rate' in name or 'Return' in name or 'Alpha' in name or 'IRR' in name:
                # 百分比格式
                print(f"{name:25s}: {value*100:8.2f}%")
            elif 'Ratio' in name or 'Beta' in name or 'Factor' in name:
                # 数值格式
                print(f"{name:25s}: {value:8.2f}")
            else:
                # 默认格式
                print(f"{name:25s}: {value:8.4f}")
        
        print("="*60)
        
        return metrics

def compare_strategies(strategy_returns_dict: Dict[str, np.ndarray], 
                      benchmark_returns: np.ndarray = None,
                      risk_free_rate: float = 0.02):
    """对比多个策略
    
    参数:
        strategy_returns_dict: {'策略名': returns数组}
        benchmark_returns: 基准收益率
        risk_free_rate: 无风险利率
    """
    results = {}
    
    for strategy_name, returns in strategy_returns_dict.items():
        metrics_calc = TradingMetrics(returns, benchmark_returns, risk_free_rate)
        results[strategy_name] = metrics_calc.get_all_metrics()
    
    # 创建对比表
    df = pd.DataFrame(results).T
    
    return df


if __name__ == "__main__":
    print("=== 测试交易指标计算 ===\n")
    
    # 生成模拟收益率
    np.random.seed(42)
    n_days = 252  # 1年
    
    # 策略1：正收益，低波动
    strategy_returns = np.random.randn(n_days) * 0.01 + 0.0005
    
    # 基准：市场收益
    benchmark_returns = np.random.randn(n_days) * 0.015 + 0.0003
    
    # 计算指标
    metrics = TradingMetrics(strategy_returns, benchmark_returns, risk_free_rate=0.02)
    metrics.print_metrics("测试策略")
    
    # 对比多个策略
    print("\n=== 策略对比 ===\n")
    
    strategy2_returns = np.random.randn(n_days) * 0.02 + 0.0008
    strategy3_returns = np.random.randn(n_days) * 0.005 + 0.0002
    
    comparison = compare_strategies({
        'Low Vol Strategy': strategy_returns,
        'High Vol Strategy': strategy2_returns,
        'Conservative Strategy': strategy3_returns
    }, benchmark_returns)
    
    print(comparison)
