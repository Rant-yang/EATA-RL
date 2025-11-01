import _thread
import threading
from contextlib import contextmanager

import numpy as np
from numpy import *
from gplearn.functions import make_function
from scipy.optimize import minimize
from sympy import simplify, expand


class TimeoutException(Exception):
    def __init__(self, msg=''):
        self.msg = msg


@contextmanager
def time_limit(seconds, msg=''):
    timer = threading.Timer(seconds, lambda: _thread.interrupt_main())
    timer.start()
    try:
        yield
    except KeyboardInterrupt:
        raise TimeoutException("Timed out for operation {}".format(msg))
    finally:
        # if the action ends in specified time, timer is canceled
        timer.cancel()


def simplify_eq(eq):
    if eq is None or not isinstance(eq, str) or eq.strip() == "":
        return "0"
    return str(expand(simplify(eq)))


def prune_poly_c(eq):
    '''
    if polynomial of C appear in eq, reduce to C for computational efficiency.
    '''
    eq = simplify_eq(eq)
    if 'C**' in eq:
        c_poly = ['C**' + str(i) for i in range(10)]
        for c in c_poly:
            if c in eq:
                eq = eq.replace(c, 'C')
    return simplify_eq(eq)


def score_with_est(eq, tree_size, data, t_limit=1.0, eta=0.999):
    if not eq or not isinstance(eq, str) or not eq.strip():
        return 0, "0"

    """
    该函数计算一个完整解析树的奖励分数。
    如果方程中包含占位符C，也会为C执行估计。
    奖励 = 1 / (1 + MSE) * Penalty ** num_term

    这是主函数的开始，它接受五个参数：eq（一个字符串，表示解析树生成的方程式），tree_size（整数，解析树的大小），
    data（二维 numpy 数组，表示用于评分的数据），t_limit（表示计算评分的时间限制）和 eta（一个用于计算惩罚因子的超参数）。

    参数:
    eq : 字符串对象，已发现的方程（包含占位符C的系数）。
    tree_size : 整数对象，完整解析树中的产生规则数。
    data : 二维numpy数组，测量数据，包括独立变量和因变量（最后一行）。
    t_limit : 浮点数对象，单次评估的时间限制（秒），默认为1秒。

    返回值:
    score: 浮点数，已发现的方程的奖励分数。
    eq: 字符串，包含估计数值的已发现方程。
    """
    
    local_vars = {
        'cos': np.cos,
        'sin': np.sin,
        'exp': np.exp,
        'log': np.log,
        'sqrt': np.sqrt,
    }

    try:
        num_var = data.shape[0] - 1
        for i in range(num_var):
            local_vars[f'x{i}'] = data[i, :]
        f_true = data[-1, :]
        local_vars['f_true'] = f_true

        c_count = eq.count('C')

        with time_limit(t_limit, 'evaluate_equation'):
            f_pred = np.array([]) # Initialize f_pred
            if c_count == 0:
                f_pred = eval(eq, {"__builtins__": None}, local_vars)
            elif c_count >= 10:
                return 0, eq
            else:
                c_lst_names = ['c' + str(i) for i in range(c_count)]
                eq_with_c_vars = eq
                for c_name in c_lst_names:
                    eq_with_c_vars = eq_with_c_vars.replace('C', c_name, 1)

                def eq_test(c_values):
                    # Create a temporary context for optimization
                    opt_vars = local_vars.copy()
                    for i in range(len(c_values)):
                        opt_vars[c_lst_names[i]] = c_values[i]
                    
                    try:
                        pred = eval(eq_with_c_vars, {"__builtins__": None}, opt_vars)
                        # Ensure pred is a numpy array
                        if not isinstance(pred, np.ndarray):
                            pred = np.repeat(pred, len(f_true))
                        
                        # Handle shape mismatch
                        if pred.shape != f_true.shape:
                            # This might happen if the expression is a constant
                            if pred.size == 1:
                                pred = np.repeat(pred, f_true.shape)
                            else: # More complex mismatch, return high error
                                return np.inf
                        
                        return np.linalg.norm(pred - f_true, 2)
                    except Exception:
                        return np.inf # Return a large error if eval fails

                x0 = [1.0] * len(c_lst_names)
                opt_result = minimize(eq_test, x0, method='Powell', tol=1e-6)
                c_lst_values = [np.round(x, 4) if abs(x) > 1e-2 else 0 for x in opt_result.x]
                
                eq_est = eq_with_c_vars
                final_eval_vars = local_vars.copy()
                for i in range(len(c_lst_values)):
                    eq_est = eq_est.replace(c_lst_names[i], str(c_lst_values[i]), 1)
                    final_eval_vars[c_lst_names[i]] = c_lst_values[i]

                eq = eq_est.replace('+-', '-')
                f_pred = eval(eq, {"__builtins__": None}, final_eval_vars)

        # Ensure f_pred is a numpy array of the correct shape
        if not isinstance(f_pred, np.ndarray) or f_pred.shape != f_true.shape:
             # Handle constants or shape mismatches
            if isinstance(f_pred, (int, float)):
                f_pred = np.repeat(f_pred, f_true.shape)
            else:
                # If it's still not matching, we can't calculate a meaningful score
                return 0, eq
        
        # Check for non-numeric or empty arrays
        if f_pred.size == 0 or not np.isfinite(f_pred).all():
            return 0, eq

        # Ensure f_pred and f_true are finite before calculating MSE
        f_pred = np.nan_to_num(f_pred, nan=0.0, posinf=0.0, neginf=0.0)
        f_true = np.nan_to_num(f_true, nan=0.0, posinf=0.0, neginf=0.0)

        mse = np.linalg.norm(f_pred - f_true, 2)**2 / f_true.shape[0]
        # Temporarily simplify the reward to focus on MSE, ignoring complexity penalty
        r = 1.0 / (1.0 + mse)

    except Exception as e:
        # Any exception in this whole process means scoring fails
        # print(f"DEBUG: Scoring failed for eq='{eq}'. Error: {e}") # Optional: for debugging
        return 0, eq

    return r, eq

