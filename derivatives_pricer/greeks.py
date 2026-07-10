"""
希腊字母（Greeks）计算模块
=============================

本模块实现Black-Scholes模型下期权希腊字母的解析公式计算，
并提供数值差分验证功能。

希腊字母是期权价格对各参数的敏感度度量：
- Delta (delta): dV/dS，期权价格对标的价格的敏感度
- Gamma (gamma): d^2V/dS^2，Delta对标价格的敏感度（凸性）
- Theta (theta): dV/dt，期权价格随时间衰减的速率
- Vega  (vega):  dV/dsigma，期权价格对波动率的敏感度
- Rho   (rho):   dV/dr，期权价格对无风险利率的敏感度

解析公式（广义BS，含连续股息收益率q）：
-----------------------------------------
对于 d1 = [ln(S/K) + (r-q+sigma^2/2)*T] / (sigma*sqrt(T))
     d2 = d1 - sigma*sqrt(T)

看涨期权:
    Delta = e^(-q*T) * N(d1)
    Gamma = e^(-q*T) * n(d1) / (S * sigma * sqrt(T))
    Theta = -S*e^(-q*T)*n(d1)*sigma/(2*sqrt(T)) - r*K*e^(-r*T)*N(d2) + q*S*e^(-q*T)*N(d1)
    Vega  = S * e^(-q*T) * n(d1) * sqrt(T)
    Rho   = K * T * e^(-r*T) * N(d2)

看跌期权:
    Delta = e^(-q*T) * (N(d1) - 1)
    Gamma = e^(-q*T) * n(d1) / (S * sigma * sqrt(T))    [与看涨相同]
    Theta = -S*e^(-q*T)*n(d1)*sigma/(2*sqrt(T)) + r*K*e^(-r*T)*N(-d2) - q*S*e^(-q*T)*N(-d1)
    Vega  = S * e^(-q*T) * n(d1) * sqrt(T)              [与看涨相同]
    Rho   = -K * T * e^(-r*T) * N(-d2)

其中 n(x) 为标准正态分布概率密度函数
注意: Theta 中的时间单位为"年"，即每经过1年Theta衰减的值
"""

import numpy as np
from scipy.stats import norm

from .black_scholes import d1, d2, bs_price, vega as bs_vega


# ============================================================================
# 各希腊字母的解析公式
# ============================================================================

def delta(S, K, T, r, sigma, option_type='call', q=0.0):
    """
    计算Delta: dV/dS

    看涨: Delta = e^(-q*T) * N(d1)
    看跌: Delta = e^(-q*T) * (N(d1) - 1) = -e^(-q*T) * N(-d1)

    参数:
        S: 标的资产价格
        K: 执行价
        T: 到期时间（年）
        r: 无风险利率
        sigma: 波动率
        option_type: 'call' 或 'put'
        q: 连续股息收益率

    返回:
        float: Delta值
    """
    if T <= 0 or sigma <= 0:
        # 到期时Delta退化为指示函数
        if option_type == 'call':
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0

    _d1 = d1(S, K, T, r, sigma, q)
    if option_type == 'call':
        return np.exp(-q * T) * norm.cdf(_d1)
    else:
        return np.exp(-q * T) * (norm.cdf(_d1) - 1.0)


def gamma(S, K, T, r, sigma, option_type='call', q=0.0):
    """
    计算Gamma: d^2V/dS^2

    Gamma = e^(-q*T) * n(d1) / (S * sigma * sqrt(T))

    注意: 看涨和看跌的Gamma相同

    参数: 同 delta()
    返回:
        float: Gamma值
    """
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0

    _d1 = d1(S, K, T, r, sigma, q)
    return np.exp(-q * T) * norm.pdf(_d1) / (S * sigma * np.sqrt(T))


def theta(S, K, T, r, sigma, option_type='call', q=0.0):
    """
    计算Theta: dV/dt（注意：这里theta是dV/dt，即时间增加时期权价值变化率）

    看涨: Theta = -S*e^(-q*T)*n(d1)*sigma/(2*sqrt(T)) - r*K*e^(-r*T)*N(d2) + q*S*e^(-q*T)*N(d1)
    看跌: Theta = -S*e^(-q*T)*n(d1)*sigma/(2*sqrt(T)) + r*K*e^(-r*T)*N(-d2) - q*S*e^(-q*T)*N(-d1)

    注意: 这里返回的是年化Theta（dV/dt，t为时间）。
    通常Theta为负值，表示期权随时间流逝而贬值。
    日历日Theta = 年化Theta / 365

    参数: 同 delta()
    返回:
        float: Theta值（年化）
    """
    if T <= 0 or sigma <= 0:
        return 0.0

    _d1 = d1(S, K, T, r, sigma, q)
    _d2 = d2(S, K, T, r, sigma, q)

    # 共同项: -S*e^(-q*T)*n(d1)*sigma/(2*sqrt(T))
    common = -S * np.exp(-q * T) * norm.pdf(_d1) * sigma / (2.0 * np.sqrt(T))

    if option_type == 'call':
        # 看涨: -r*K*e^(-r*T)*N(d2) + q*S*e^(-q*T)*N(d1)
        return common - r * K * np.exp(-r * T) * norm.cdf(_d2) + q * S * np.exp(-q * T) * norm.cdf(_d1)
    else:
        # 看跌: +r*K*e^(-r*T)*N(-d2) - q*S*e^(-q*T)*N(-d1)
        return common + r * K * np.exp(-r * T) * norm.cdf(-_d2) - q * S * np.exp(-q * T) * norm.cdf(-_d1)


def vega_greek(S, K, T, r, sigma, option_type='call', q=0.0):
    """
    计算Vega: dV/dsigma

    Vega = S * e^(-q*T) * n(d1) * sqrt(T)

    注意: 看涨和看跌的Vega相同

    参数: 同 delta()
    返回:
        float: Vega值
    """
    # 复用black_scholes模块中的vega函数
    return bs_vega(S, K, T, r, sigma, q)


def rho(S, K, T, r, sigma, option_type='call', q=0.0):
    """
    计算Rho: dV/dr

    看涨: Rho = K * T * e^(-r*T) * N(d2)
    看跌: Rho = -K * T * e^(-r*T) * N(-d2)

    参数: 同 delta()
    返回:
        float: Rho值
    """
    if T <= 0 or sigma <= 0:
        return 0.0

    _d2 = d2(S, K, T, r, sigma, q)

    if option_type == 'call':
        return K * T * np.exp(-r * T) * norm.cdf(_d2)
    else:
        return -K * T * np.exp(-r * T) * norm.cdf(-_d2)


# ============================================================================
# 汇总计算
# ============================================================================

def all_greeks(S, K, T, r, sigma, option_type='call', q=0.0):
    """
    一次性计算所有希腊字母

    参数: 同 delta()
    返回:
        dict: 包含所有希腊字母的字典
            - 'delta': Delta值
            - 'gamma': Gamma值
            - 'theta': Theta值（年化）
            - 'vega': Vega值
            - 'rho': Rho值
    """
    return {
        'delta': delta(S, K, T, r, sigma, option_type, q),
        'gamma': gamma(S, K, T, r, sigma, option_type, q),
        'theta': theta(S, K, T, r, sigma, option_type, q),
        'vega': vega_greek(S, K, T, r, sigma, option_type, q),
        'rho': rho(S, K, T, r, sigma, option_type, q),
    }


# ============================================================================
# 数值差分验证
# ============================================================================

def numerical_greeks(S, K, T, r, sigma, option_type='call', q=0.0,
                     h_S=None, h_sigma=None, h_r=None, h_t=None):
    """
    使用数值差分法计算希腊字母，用于验证解析公式的正确性

    采用中心差分（二阶精度）:
        f'(x) ≈ [f(x+h) - f(x-h)] / (2h)
        f''(x) ≈ [f(x+h) - 2f(x) + f(x-h)] / h^2

    参数:
        S, K, T, r, sigma, option_type, q: 同前
        h_S: 标的价格差分步长（默认 S*0.01）
        h_sigma: 波动率差分步长（默认 0.0001）
        h_r: 利率差分步长（默认 0.0001）
        h_t: 时间差分步长（默认 0.0001）

    返回:
        dict: 数值差分计算的希腊字母
    """
    # 设置默认差分步长
    if h_S is None:
        h_S = S * 0.01
    if h_sigma is None:
        h_sigma = 1e-4
    if h_r is None:
        h_r = 1e-4
    if h_t is None:
        h_t = 1e-4

    def price(S_p, K_p, T_p, r_p, sigma_p):
        return bs_price(S_p, K_p, T_p, r_p, sigma_p, option_type, q)

    # Delta: dV/dS（中心差分）
    delta_num = (price(S + h_S, K, T, r, sigma) - price(S - h_S, K, T, r, sigma)) / (2 * h_S)

    # Gamma: d^2V/dS^2（中心差分）
    gamma_num = (price(S + h_S, K, T, r, sigma) - 2 * price(S, K, T, r, sigma)
                 + price(S - h_S, K, T, r, sigma)) / (h_S ** 2)

    # Vega: dV/dsigma（中心差分）
    vega_num = (price(S, K, T, r, sigma + h_sigma) - price(S, K, T, r, sigma - h_sigma)) / (2 * h_sigma)

    # Rho: dV/dr（中心差分）
    rho_num = (price(S, K, T, r + h_r, sigma) - price(S, K, T, r - h_r, sigma)) / (2 * h_r)

    # Theta: dV/dt = -dV/dT（注意：Theta = dV/dt，而T是到期时间，t是当前时间，dt = -dT）
    # 数值差分: dV/dT ≈ [V(T+h) - V(T-h)] / (2h)
    # 因此 Theta = -dV/dT = [V(T-h) - V(T+h)] / (2h)
    theta_num = (price(S, K, T - h_t, r, sigma) - price(S, K, T + h_t, r, sigma)) / (2 * h_t)

    return {
        'delta': delta_num,
        'gamma': gamma_num,
        'theta': theta_num,
        'vega': vega_num,
        'rho': rho_num,
    }


# ============================================================================
# 希腊字母可视化数据表
# ============================================================================

def greeks_table(S_range, K, T, r, sigma, option_type='call', q=0.0):
    """
    生成希腊字母对标的价格的数据表，用于可视化

    参数:
        S_range (array-like): 标的价格数组
        K, T, r, sigma, option_type, q: 同前

    返回:
        dict: 包含标的价格和对应希腊字母的字典
    """
    S_arr = np.asarray(S_range)

    deltas = np.array([delta(s, K, T, r, sigma, option_type, q) for s in S_arr])
    gammas = np.array([gamma(s, K, T, r, sigma, option_type, q) for s in S_arr])
    thetas = np.array([theta(s, K, T, r, sigma, option_type, q) for s in S_arr])
    vegas = np.array([vega_greek(s, K, T, r, sigma, option_type, q) for s in S_arr])
    rhos = np.array([rho(s, K, T, r, sigma, option_type, q) for s in S_arr])

    return {
        'S': S_arr,
        'delta': deltas,
        'gamma': gammas,
        'theta': thetas,
        'vega': vegas,
        'rho': rhos,
    }
