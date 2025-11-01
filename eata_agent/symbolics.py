import numpy as np
from gplearn.functions import make_function

balldrop_exp = ['Baseball',
                'Blue Basketball',
                'Green Basketball',
                'Volleyball',
                'Bowling Ball',
                'Golf Ball',
                'Tennis Ball',
                'Whiffle Ball 1',
                'Whiffle Ball 2',
                'Yellow Whiffle Ball',
                'Orange Whiffle Ball']

## 金融时间序列专用函数 - 兼容gplearn的向量化实现

# gplearn配置函数必须接受数组并返回数组
# 定义简化版金融函数，只用于文法生成

def _protected_exponent(x):
    """Exp函数的安全实现版本"""
    with np.errstate(over='ignore'):
        return np.where(x < 100, np.exp(x), np.exp(100))

# 安全除法函数
# 这个已经在gplearn中定义了所以不需要重复定义
def _protected_division(x1, x2):
    """Divides the first input by the second, returning 1 when the second input is
    close to zero."""
    return np.where(np.abs(x2) > 0.001, np.divide(x1, x2), 1.)

# 安全对数
# 这个在gplearn中也有定义
def _protected_log(x1):
    """Returns the natural logarithm of the absolute value of the input,
    returning 0 for zero or negative inputs."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(x1 > 0.001, np.log(np.abs(x1)), 0.)

# 定义金融指标函数 - 引入默认常数值

def _protected_delay(x1, x2):
    """延迟函数 - 安全实现版本，如5天滞后"""
    # 使用固定延迟而不是可变参数，因为gplearn会将一个算式作为这个参数
    delay_days = 5  # 固定值
    return np.roll(x1, delay_days)

def _protected_ma(x1, x2):
    """移动平均 - 安全实现版本，默认10日移动平均"""
    # 固定窗口大小
    window_size = 10
    result = np.zeros_like(x1)
    for i in range(len(x1)):
        if i < window_size-1:
            result[i] = np.mean(x1[:i+1])
        else:
            result[i] = np.mean(x1[i-window_size+1:i+1])
    return result

def _protected_diff(x1, x2):
    """差分函数 - 安全实现版本，默认1日差分"""
    return np.concatenate([[0], x1[1:] - x1[:-1]])

def _protected_max_n(x1, x2):
    """最大值函数 - 安全实现版本，默认5日最大值"""
    window_size = 5
    result = np.zeros_like(x1)
    for i in range(len(x1)):
        if i < window_size-1:
            result[i] = np.max(x1[:i+1])
        else:
            result[i] = np.max(x1[i-window_size+1:i+1])
    return result

def _protected_min_n(x1, x2):
    """最小值函数 - 安全实现版本，默认5日最小值"""
    window_size = 5
    result = np.zeros_like(x1)
    for i in range(len(x1)):
        if i < window_size-1:
            result[i] = np.min(x1[:i+1])
        else:
            result[i] = np.min(x1[i-window_size+1:i+1])
    return result

def _protected_mom(x1, x2):
    """动量函数 - 安全实现版本，默认5日动量"""
    lag = 5
    result = np.zeros_like(x1)
    result[lag:] = x1[lag:] / np.where(np.abs(x1[:-lag]) > 0.001, x1[:-lag], 0.001) - 1
    return result

def _protected_rsi(x1, x2):
    """相对强弱指标 - 安全实现版本，默认14日RSI"""
    period = 14
    # 计算变化的差分
    delta = np.zeros_like(x1)
    delta[1:] = x1[1:] - x1[:-1]
    
    # 获取正负值
    gain = np.maximum(delta, 0)
    loss = -np.minimum(delta, 0)
    
    # 预填充结果数组
    result = np.zeros_like(x1)
    
    # 使用简化的版本: 平均上涨除以平均下跌
    avg_gain = np.zeros_like(x1)
    avg_loss = np.zeros_like(x1)
    
    # 使用滚动窗口计算
    for i in range(period, len(x1)):
        avg_gain[i] = np.mean(gain[i-period+1:i+1])
        avg_loss[i] = np.mean(loss[i-period+1:i+1])
        
        if avg_loss[i] == 0:
            result[i] = 100
        else:
            rs = avg_gain[i] / np.maximum(avg_loss[i], 0.001)
            result[i] = 100 - (100 / (1 + rs))
    
    return result

def _protected_volatility(x1, x2):
    """波动率函数 - 安全实现版本，默认20日波动率"""
    window_size = 20
    result = np.zeros_like(x1)
    for i in range(len(x1)):
        if i < window_size-1:
            segment = x1[:i+1]
        else:
            segment = x1[i-window_size+1:i+1]
        result[i] = np.std(segment) / np.maximum(np.mean(segment), 0.001)
    return result

def _protected_ite(x1, x2, x3):
    """条件函数 - 如果x1>0，返回x2，否则返回x3"""
    return np.where(x1 > 0, x2, x3)

## 加强版金融时间序列文法
def gen_enhanced_finance_grammar(n, lookBACK):
    """结合基础数学运算和金融专用函数的增强算子集"""
    # 原始变量映射
    terminals = [f'A->x{i}' for i in range(n*lookBACK)]
    
    # 基础算术运算
    basic_ops = [
        'A->A+A', 'A->A-A', 'A->A*A', 'A->A/A', 
        'A->A*C', 'A->A+C', 'A->C-A',  # 带常数的运算
    ]
    
    # 数学函数
    math_funcs = [
        'A->cos(A)', 'A->sin(A)', 'A->exp(A)', 'A->log(B)', 'A->sqrt(B)',
        'A->abs(A)', 'A->A**2', 'A->A**3', 'A->A**0.5',  # 幂运算和绝对值
    ]
    
    # 金融专用时间序列函数（核心部分）
    finance_funcs = [
        # 时间滞后项 - 捕捉历史信息
        'A->delay(A,1)',   # t-1时刻的值
        'A->delay(A,5)',   # t-5时刻的值
        
        # 移动平均 - 平滑趋势
        'A->ma(A,5)',      # 5日移动平均
        'A->ma(A,10)',     # 10日移动平均
        'A->ma(A,20)',     # 20日移动平均
        
        # 差分和变化率 - 捕捉变化
        'A->diff(A,1)',    # 一阶差分 (t值-t-1值)
        'A->diff(A,5)',    # 五日差分
        'A->mom(A,10)',    # 动量 (当前值/10天前值)
        
        # 极值函数 - 捕捉波动边界
        'A->max_n(A,5)',   # 5日最大值
        'A->min_n(A,5)',   # 5日最小值
        'A->min(A,B)',     # 取两值最小值
        'A->max(A,B)',     # 取两值最大值
        
        # 技术指标 - 股票特有指标
        'A->rsi(A,14)',    # 相对强弱指标
        'A->volatility(A,20)',  # 波动率
    ]
    
    # 辅助非终结符和条件操作
    helper_rules = [
        'B->B+B', 'B->B-B', 'B->A', 'B->abs(A)', 'B->1', 'B->2', 
        'B->ma(A,10)', 'B->A**2', 'B->max(A,0)', 'B->min(A,1)'
    ]
    
    # 简单的逻辑操作
    logic_ops = [
        'A->ite(D,A,A)',  # if-then-else结构
        'D->A>B', 'D->A<B'  # 简单比较
    ]
    
    return basic_ops + math_funcs + finance_funcs + helper_rules + logic_ops + terminals

## production rules for each benchmark for SPL
rule_map = {
    # 新增：金融时间序列预测专用规则
    'finance': gen_enhanced_finance_grammar(15, 10),  # 默认参数，实际使用时会根据真实特征数和lookBACK重新生成
    
    'nguyen-1': ['A->A+A', 'A->A-A', 'A->A*A', 'A->A/A',
                 'A->x', 'A->x**2', 'A->x**4',
                 'A->exp(A)', 'A->cos(x)', 'A->sin(x)'],

    'nguyen-2': ['A->A+A', 'A->A-A', 'A->A*A', 'A->A/A',
                 'A->x', 'A->x**2', 'A->x**4',
                 'A->exp(A)', 'A->cos(x)', 'A->sin(x)'],

    'nguyen-3': ['A->A+A', 'A->A-A', 'A->A*A', 'A->A/A',
                 'A->x', 'A->x**2', 'A->x**4',
                 'A->exp(A)', 'A->cos(x)', 'A->sin(x)'],

    'nguyen-4': ['A->A+A', 'A->A-A', 'A->A*A', 'A->A/A',
                 'A->x', 'A->x**2', 'A->x**4',
                 'A->exp(A)', 'A->cos(x)', 'A->sin(x)'],

    'nguyen-5': ['A->A+A', 'A->A-A', 'A->A*A', 'A->A/A', 'A->exp(A)',
                 'A->cos(B)', 'A->sin(B)', 'A->1', 'A->x',
                 'B->B+B', 'B->B-B', 'B->x', 'B->x**2', 'B->x**3', 'B->1'],

    'nguyen-6': ['A->A+A', 'A->A-A', 'A->A*A', 'A->A/A', 'A->exp(A)',
                 'A->cos(B)', 'A->sin(B)', 'A->1', 'A->x',
                 'B->B+B', 'B->B-B', 'B->x', 'B->x**2', 'B->x**3', 'B->1'],

    'nguyen-7': ['A->A+A', 'A->A-A', 'A->A*A', 'A->A/A', 'A->exp(A)',
                 'A->x', 'A->cos(x)', 'A->sin(x)', 'A->log(B)', 'A->sqrt(B)',
                 'B->B+B', 'B->B-B', 'B->x', 'B->x**2', 'B->x**3', 'B->1'],

    'nguyen-8': ['A->A+A', 'A->A-A', 'A->A*A', 'A->A/A',
                 'A->x', 'A->cos(A)', 'A->sin(A)', 'A->exp(A)',
                 'A->log(A)', 'A->sqrt(A)'],

    'nguyen-9': ['A->A+A', 'A->A-A', 'A->A*A', 'A->A/A', 'A->exp(A)',
                 'A->1', 'A->x', 'A->y', 'A->cos(B)', 'A->sin(B)', 'B->B+B', 'B->B-B',
                 'B->1', 'B->x', 'B->y', 'B->x**2', 'B->y**2', 'B->x*y',
                 'B->x**2*y', 'B->x*y**2', 'B->x**3', 'B->y**3'],

    'nguyen-10': ['A->A+A', 'A->A-A', 'A->A*A', 'A->A/A', 'A->exp(A)',
                  'A->1', 'A->x', 'A->y', 'A->cos(B)', 'A->sin(B)', 'B->B+B', 'B->B-B',
                  'B->1', 'B->x', 'B->y', 'B->x**2', 'B->y**2', 'B->x*y',
                  'B->x**2*y', 'B->x*y**2', 'B->x**3', 'B->y**3'],

    'nguyen-11': ['A->x', 'A->y', 'A->A+A', 'A->A-A', 'A->A*A', 'A->A/A',
                  'A->exp(A)', 'A->log(B)', 'A->sqrt(B)', 'A->cos(B)', 'A->sin(B)',
                  'B->B+B', 'B->B-B', 'B->1', 'B->x', 'B->y', 'B->x**2', 'B->y**2', 'B->x*y',
                  'B->x**2*y', 'B->x*y**2', 'B->x**3', 'B->y**3'],

    'nguyen-12': ['A->(A+A)', 'A->(A-A)', 'A->A*A', 'A->A/A',
                  'A->x', 'A->x**2', 'A->x**4', 'A->y', 'A->y**2', 'A->y**4',
                  'A->1', 'A->2', 'A->exp(A)',
                  'A->cos(x)', 'A->sin(x)', 'A->cos(y)', 'A->sin(y)'],

    'nguyen-1c': ['A->A+A', 'A->A-A', 'A->A*A', 'A->A/A', 'A->x', 'A->x**2', 'A->x**4',
                  'A->A*C', 'A->exp(x)', 'A->cos(C*x)', 'A->sin(C*x)'],

    'nguyen-2c': ['A->A+A', 'A->A-A', 'A->A*A', 'A->A/A', 'A->x', 'A->x**2', 'A->x**4',
                  'A->A*C', 'A->exp(x)', 'A->cos(C*x)', 'A->sin(C*x)'],

    'nguyen-5c': ['A->A+A', 'A->A-A', 'A->A*A', 'A->A/A', 'A->exp(A)',
                  'A->cos(B)', 'A->sin(B)', 'A->1', 'A->x', 'A->A*C',
                  'B->B+B', 'B->B-B', 'B->x', 'B->x**2', 'B->x**3', 'B->1', 'B->B*C'],

    'nguyen-7c': ['A->A+A', 'A->A-A', 'A->A*A', 'A->A/A', 'A->exp(A)', 'A->A*C',
                  'A->x', 'A->cos(x)', 'A->sin(x)', 'A->log(B)', 'A->sqrt(B)',
                  'B->B+B', 'B->B-B', 'B->x', 'B->x**2', 'B->x**3', 'B->1', 'B->B*C'],

    'nguyen-8c': ['A->A+A', 'A->A-A', 'A->A*A', 'A->A/A',
                  'A->x', 'A->cos(A)', 'A->sin(A)', 'A->exp(A)', 'A->A*C',
                  'A->log(A)', 'A->sqrt(A)'],

    'nguyen-9c': ['A->A+A', 'A->A-A', 'A->A*A', 'A->A/A', 'A->A*C',
                  'A->1', 'A->x', 'A->y', 'A->cos(B)', 'A->sin(B)', 'A->exp(B)',
                  'B->B*C', 'B->1', 'B->B+B', 'B->B-B',
                  'B->x', 'B->y', 'B->x**2', 'B->y**2', 'B->x*y',
                  'B->x**2*y', 'B->x*y**2', 'B->x**3', 'B->y**3'],

    **dict.fromkeys(balldrop_exp, ['A->A+A', 'A->A-A', 'A->A*A', 'A->A/A', 'A->A*C',
                                   'A->1', 'A->x', 'A->x*x', 'A->x*x*x',
                                   'A->exp(A)',
                                   'A->log(C*cosh(A))']),

    **dict.fromkeys(['dp_f1', 'dp_f2'],
                    ['A->C*wdot*cos(x1-x2)', 'A->A+A', 'A->A*A', 'A->C*A',
                     'A->W', 'W->w1', 'W->w2', 'W->wdot', 'W->W*W',
                     'A->cos(T)', 'A->sin(T)', 'T->x1', 'T->x2', 'T->T+T', 'T->T-T',
                     'A->sign(S)', 'S->w1', 'S->w2', 'S->wdot', 'A->S+S', 'B->S-S']),

    **dict.fromkeys(['lorenz_x', 'lorenz_y', 'lorenz_z'],
                    ['A->A+A', 'A->A-A', 'A->A*A', 'A->A/A', 'A->A*C',
                     'A->x', 'A->y', 'A->z']),

    'NEMoTS': [], # 动态生成，由model.py负责

    # 'NEMoTS': [
    #     'A->A+A', 'A->A-A', 'A->A*A', 'A->A/A',  # 基本算术运算
    #     'A->x', 'A->cos(A)', 'A->sin(A)', 'A->exp(A)', 'A->A*C',  # 常见函数和操作
    #     'A->log(A)', 'A->sqrt(A)',
    #     'A->tan(A)',  # 正切函数
    #     'A->cot(A)',  # 余切函数
    #     'A->sec(A)',  # 正割函数
    #     'A->csc(A)',  # 余割函数
    #     'A->A^2',  # 平方
    #     'A->A^3',  # 立方
    #     'A->A^n',  # 幂运算
    #     'A->|A|',  # 绝对值
    #     'A->asin(A)',  # 反正弦函数
    #     'A->acos(A)',  # 反余弦函数
    #     'A->atan(A)',  # 反正切函数
    #     'A->acot(A)',  # 反余切函数
    #     'A->ln(A)',  # 自然对数
    #     'A->log10(A)',  # 以10为底的对数
    #     'A->logb(A, B)',  # 以B为底A的对数
    #     'A->e^A',  # e的A次幂
    # ]
}

## non-terminal nodes for each task for SPL
ntn_map = {
    'nguyen-1': ['A'],
    'nguyen-2': ['A'],
    'nguyen-3': ['A'],
    'nguyen-4': ['A'],
    'nguyen-5': ['A', 'B'],
    'nguyen-6': ['A', 'B'],
    'nguyen-7': ['A', 'B'],
    'nguyen-8': ['A'],
    'nguyen-9': ['A', 'B'],
    'nguyen-10': ['A', 'B'],
    'nguyen-11': ['A', 'B'],
    'nguyen-12': ['A'],
    'nguyen-1c': ['A'],
    'nguyen-2c': ['A'],
    'nguyen-5c': ['A', 'B'],
    'nguyen-7c': ['A', 'B'],
    'nguyen-8c': ['A'],
    'nguyen-9c': ['A', 'B'],
    **dict.fromkeys(balldrop_exp, ['A']),
    **dict.fromkeys(['dp_f1', 'dp_f2'], ['A', 'W', 'T', 'S']),
    **dict.fromkeys(['lorenz_x', 'lorenz_y', 'lorenz_z'], ['A']),

    'NEMoTS': ['A']
}


## function set for GP


def _protected_exponent(x):
    with np.errstate(over='ignore'):
        return np.where(np.abs(x) < 100, np.exp(x), 0.)


exponential = make_function(function=_protected_exponent, name='exp', arity=1)

# 将自定义互操作加入函数的映射表
# 自定义工具函数
from gplearn.functions import make_function

# 将新增函数定义为gplearn格式 - 编译函数
# 使用保护版实现的函数
delay_func = make_function(function=_protected_delay, name='delay', arity=2)
ma_func = make_function(function=_protected_ma, name='ma', arity=2)
diff_func = make_function(function=_protected_diff, name='diff', arity=2)
max_n_func = make_function(function=_protected_max_n, name='max_n', arity=2)
min_n_func = make_function(function=_protected_min_n, name='min_n', arity=2)
mom_func = make_function(function=_protected_mom, name='mom', arity=2)
rsi_func = make_function(function=_protected_rsi, name='rsi', arity=2)
volatility_func = make_function(function=_protected_volatility, name='volatility', arity=2)
ite_func = make_function(function=_protected_ite, name='ite', arity=3)

f_set = {
    # 金融时间序列函数集
    'finance': ("add", "sub", "mul", "div", "sin", "cos", exponential, "log", "sqrt", 
               delay_func, ma_func, diff_func, max_n_func, min_n_func, mom_func, 
               rsi_func, volatility_func, ite_func),
    
    # 原有函数集
    'nguyen-1': ("add", "sub", "mul", "div", "sin", "cos", exponential),
    'nguyen-2': ("add", "sub", "mul", "div", "sin", "cos", exponential),
    'nguyen-3': ("add", "sub", "mul", "div", "sin", "cos", exponential),
    'nguyen-4': ("add", "sub", "mul", "div", "sin", "cos", exponential),
    'nguyen-5': ("add", "sub", "mul", "div", "sin", "cos", exponential),
    'nguyen-6': ("add", "sub", "mul", "div", "sin", "cos", exponential),
    'nguyen-7': ("add", "sub", "mul", "div", "sin", "cos", exponential, "log", "sqrt"),
    'nguyen-8': ("add", "sub", "mul", "div", "sin", "cos", exponential, "log", "sqrt"),
    'nguyen-9': ("add", "sub", "mul", "div", "sin", "cos", exponential),
    'nguyen-10': ("add", "sub", "mul", "div", "sin", "cos", exponential),
    'nguyen-11': ("add", "sub", "mul", "div", "sin", "cos", exponential, "log", "sqrt"),
    'nguyen-12': ("add", "sub", "mul", "div", "sin", "cos", exponential),
    'nguyen-1c': ("add", "sub", "mul", "div", "sin", "cos", exponential),
    'nguyen-2c': ("add", "sub", "mul", "div", "sin", "cos", exponential),
    'nguyen-5c': ("add", "sub", "mul", "div", "sin", "cos", exponential),
    'nguyen-8c': ("add", "sub", "mul", "div", "sin", "cos", exponential, "log", "sqrt"),
    'nguyen-9c': ("add", "sub", "mul", "div", "sin", "cos", exponential)
}
