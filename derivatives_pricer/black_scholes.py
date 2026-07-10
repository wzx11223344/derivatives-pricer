"""
Black-Scholes 期权定价模型（解析解）
=====================================

本模块实现以下功能：
1. 欧式看涨/看跌期权定价（支持连续股息收益率q的广义BS模型）
2. 隐含波动率反推（Newton-Raphson 迭代法）
3. Put-Call Parity 验证

广义Black-Scholes公式（Merton 1973）：
-----------------------------------------
对于标的资产价格S、执行价K、到期时间T、无风险利率r、波动率sigma、
连续股息收益率q的欧式期权：

看涨期权: C = S * e^(-q*T) * N(d1) - K * e^(-r*T) * N(d2)
看跌期权: P = K * e^(-r*T) * N(-d2) - S * e^(-q*T) * N(-d1)

其中:
    d1 = [ln(S/K) + (r - q + sigma^2/2) * T] / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

N(x) 为标准正态分布累积分布函数
"""

import numpy as np
from scipy.stats import norm


# ============================================================================
# 核心参数计算
# ============================================================================

def d1(S, K, T, r, sigma, q=0.0):
    """
    计算Black-Scholes模型中的d1参数

    参数:
        S (float): 标的资产当前价格
        K (float): 期权执行价（行权价）
        T (float): 到期时间（年化，如0.25表示3个月）
        r (float): 无风险利率（年化，如0.03表示3%）
        sigma (float): 标的资产波动率（年化，如0.2表示20%）
        q (float): 连续股息收益率（年化，默认0表示无股息）

    返回:
        float: d1值
    """
    return (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))


def d2(S, K, T, r, sigma, q=0.0):
    """
    计算Black-Scholes模型中的d2参数

    d2 = d1 - sigma * sqrt(T)

    参数: 同 d1()
    返回:
        float: d2值
    """
    return d1(S, K, T, r, sigma, q) - sigma * np.sqrt(T)


# ============================================================================
# 欧式期权定价
# ============================================================================

def bs_call(S, K, T, r, sigma, q=0.0):
    """
    Black-Scholes 欧式看涨期权定价（广义BS，支持连续股息收益率q）

    公式: C = S * e^(-q*T) * N(d1) - K * e^(-r*T) * N(d2)

    参数:
        S: 标的资产价格
        K: 执行价
        T: 到期时间（年）
        r: 无风险利率
        sigma: 波动率
        q: 连续股息收益率（默认0）

    返回:
        float: 看涨期权价格
    """
    # 边界情况处理：到期时直接返回内在价值
    if T <= 0:
        return max(S - K, 0.0)
    # 波动率为0时，期权价值为折现后的确定性现金流
    if sigma <= 0:
        return max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)

    _d1 = d1(S, K, T, r, sigma, q)
    _d2 = d2(S, K, T, r, sigma, q)

    # C = S * e^(-q*T) * N(d1) - K * e^(-r*T) * N(d2)
    return S * np.exp(-q * T) * norm.cdf(_d1) - K * np.exp(-r * T) * norm.cdf(_d2)


def bs_put(S, K, T, r, sigma, q=0.0):
    """
    Black-Scholes 欧式看跌期权定价（广义BS，支持连续股息收益率q）

    公式: P = K * e^(-r*T) * N(-d2) - S * e^(-q*T) * N(-d1)

    参数: 同 bs_call()
    返回:
        float: 看跌期权价格
    """
    if T <= 0:
        return max(K - S, 0.0)
    if sigma <= 0:
        return max(K * np.exp(-r * T) - S * np.exp(-q * T), 0.0)

    _d1 = d1(S, K, T, r, sigma, q)
    _d2 = d2(S, K, T, r, sigma, q)

    # P = K * e^(-r*T) * N(-d2) - S * e^(-q*T) * N(-d1)
    return K * np.exp(-r * T) * norm.cdf(-_d2) - S * np.exp(-q * T) * norm.cdf(-_d1)


def bs_price(S, K, T, r, sigma, option_type='call', q=0.0):
    """
    统一BS定价接口

    参数:
        S: 标的资产价格
        K: 执行价
        T: 到期时间（年）
        r: 无风险利率
        sigma: 波动率
        option_type: 'call' 或 'put'
        q: 连续股息收益率

    返回:
        float: 期权价格
    """
    if option_type.lower() == 'call':
        return bs_call(S, K, T, r, sigma, q)
    elif option_type.lower() == 'put':
        return bs_put(S, K, T, r, sigma, q)
    else:
        raise ValueError(f"不支持的期权类型: {option_type}，请使用 'call' 或 'put'")


# ============================================================================
# Vega 计算（隐含波动率反推所需）
# ============================================================================

def vega(S, K, T, r, sigma, q=0.0):
    """
    计算Vega（期权价格对波动率的一阶导数）

    公式: Vega = S * e^(-q*T) * n(d1) * sqrt(T)

    其中 n(x) 为标准正态分布概率密度函数

    参数: 同 bs_call()
    返回:
        float: Vega值
    """
    if T <= 0 or sigma <= 0:
        return 0.0
    _d1 = d1(S, K, T, r, sigma, q)
    return S * np.exp(-q * T) * norm.pdf(_d1) * np.sqrt(T)


# ============================================================================
# 隐含波动率反推
# ============================================================================

def implied_vol(price, S, K, T, r, option_type='call', q=0.0,
                tol=1e-8, max_iter=100, initial_guess=0.2):
    """
    隐含波动率反推（Newton-Raphson 迭代法）

    给定市场价格反推隐含波动率。Newton-Raphson迭代公式：
        sigma_{n+1} = sigma_n - (BS(sigma_n) - market_price) / vega(sigma_n)

    参数:
        price (float): 市场观测到的期权价格
        S: 标的资产价格
        K: 执行价
        T: 到期时间（年）
        r: 无风险利率
        option_type: 'call' 或 'put'
        q: 连续股息收益率
        tol (float): 收敛容差
        max_iter (int): 最大迭代次数
        initial_guess (float): 初始波动率猜测值

    返回:
        float: 隐含波动率
    """
    sigma = initial_guess

    for i in range(max_iter):
        # 计算当前波动率下的BS价格
        bs = bs_price(S, K, T, r, sigma, option_type, q)
        diff = bs - price

        # 检查是否收敛
        if abs(diff) < tol:
            return sigma

        # 计算Vega用于更新
        v = vega(S, K, T, r, sigma, q)
        if v < 1e-12:
            # Vega过小，无法继续迭代（深度价内/价外期权）
            break

        # Newton-Raphson 更新
        sigma = sigma - diff / v

        # 确保波动率保持正值
        if sigma <= 0:
            sigma = 0.001

    return sigma


# ============================================================================
# Put-Call Parity 验证
# ============================================================================

def put_call_parity_check(S, K, T, r, sigma, q=0.0):
    """
    Put-Call Parity 验证

    广义Put-Call Parity关系：
        C - P = S * e^(-q*T) - K * e^(-r*T)

    其中 C 为看涨期权价格，P 为看跌期权价格

    参数:
        S: 标的资产价格
        K: 执行价
        T: 到期时间
        r: 无风险利率
        sigma: 波动率
        q: 连续股息收益率

    返回:
        dict: 包含左右两边值和差值
            - 'parity_left': C - P（等式左边）
            - 'parity_right': S*e^(-q*T) - K*e^(-r*T)（等式右边）
            - 'difference': 绝对差值
    """
    C = bs_call(S, K, T, r, sigma, q)
    P = bs_put(S, K, T, r, sigma, q)

    # C - P = S*e^(-q*T) - K*e^(-r*T)
    left = C - P
    right = S * np.exp(-q * T) - K * np.exp(-r * T)

    return {
        'call_price': C,
        'put_price': P,
        'parity_left': left,
        'parity_right': right,
        'difference': abs(left - right)
    }
