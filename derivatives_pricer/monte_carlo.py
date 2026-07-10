"""
蒙特卡洛模拟期权定价模块
=============================

本模块实现以下功能：
1. 标准蒙特卡洛模拟（GBM几何布朗运动）—— 欧式期权
2. 对偶变量法（Antithetic Variates）—— 降低方差，提高效率
3. 控制变量法（Control Variates）—— 利用已知解析解降低方差
4. 最小二乘蒙特卡洛（Longstaff-Schwartz）—— 美式期权定价
5. 路径依赖期权（亚式期权：算术平均/几何平均）
6. 收敛性分析 —— 不同模拟次数的定价误差

几何布朗运动（GBM）模型:
-------------------------
    dS_t = (r - q) * S_t * dt + sigma * S_t * dW_t

精确解（对数正态分布）:
    S_T = S_0 * exp((r - q - sigma^2/2) * T + sigma * W_T)

其中 W_T ~ N(0, T)，即 sigma*sqrt(T)*Z，Z~N(0,1)

注意: 所有随机数生成使用 np.random.RandomState(random_state) 保证可复现
"""

import numpy as np
from scipy.stats import norm

from .black_scholes import bs_price

# 全局随机种子，保证蒙特卡洛结果可复现
RANDOM_STATE = 42


# ============================================================================
# GBM 路径模拟
# ============================================================================

def simulate_gbm_paths(S0, T, r, sigma, n_paths, n_steps, q=0.0,
                       random_state=RANDOM_STATE):
    """
    模拟几何布朗运动（GBM）价格路径

    使用精确解方法（对数正态分布）生成路径：
        S_{t+dt} = S_t * exp((r - q - sigma^2/2)*dt + sigma*sqrt(dt)*Z)

    参数:
        S0 (float): 初始价格
        T (float): 总时间（年）
        r (float): 无风险利率
        sigma (float): 波动率
        n_paths (int): 模拟路径数
        n_steps (int): 时间步数
        q (float): 连续股息收益率
        random_state (int): 随机种子

    返回:
        ndarray: 形状 (n_paths, n_steps+1) 的价格路径数组
                 第0列为S0，第n_steps列为S_T
    """
    rng = np.random.RandomState(random_state)
    dt = T / n_steps
    drift = (r - q - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt)

    # 生成标准正态随机数矩阵 (n_paths, n_steps)
    Z = rng.standard_normal((n_paths, n_steps))

    # 逐步构建价格路径
    S = np.zeros((n_paths, n_steps + 1))
    S[:, 0] = S0
    for t in range(1, n_steps + 1):
        S[:, t] = S[:, t - 1] * np.exp(drift + diffusion * Z[:, t - 1])

    return S


def simulate_terminal_prices(S0, T, r, sigma, n_paths, q=0.0,
                              random_state=RANDOM_STATE):
    """
    仅模拟到期时标的资产价格（不生成完整路径，效率更高）

    S_T = S_0 * exp((r - q - sigma^2/2)*T + sigma*sqrt(T)*Z)

    参数: 同 simulate_gbm_paths（但不需要 n_steps）
    返回:
        ndarray: 形状 (n_paths,) 的到期价格数组
    """
    rng = np.random.RandomState(random_state)
    Z = rng.standard_normal(n_paths)
    ST = S0 * np.exp((r - q - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)
    return ST


# ============================================================================
# 1. 标准蒙特卡洛 - 欧式期权
# ============================================================================

def mc_european(S0, K, T, r, sigma, option_type='call', q=0.0,
                n_paths=100000, random_state=RANDOM_STATE):
    """
    标准蒙特卡洛模拟 - 欧式期权定价

    方法:
        1. 模拟 n_paths 条到期价格 S_T
        2. 计算每条路径的期权收益 payoff = max(S_T - K, 0)（看涨）或 max(K - S_T, 0)（看跌）
        3. 折现取平均: V = e^(-r*T) * mean(payoff)

    参数:
        S0: 标的资产初始价格
        K: 执行价
        T: 到期时间（年）
        r: 无风险利率
        sigma: 波动率
        option_type: 'call' 或 'put'
        q: 连续股息收益率
        n_paths: 模拟路径数
        random_state: 随机种子（保证可复现）

    返回:
        dict: 包含以下字段:
            - 'price': 期权价格估计
            - 'std_error': 标准误差
            - 'confidence_interval_95': 95%置信区间
            - 'n_paths': 模拟路径数
    """
    ST = simulate_terminal_prices(S0, T, r, sigma, n_paths, q, random_state)

    # 计算收益
    if option_type == 'call':
        payoff = np.maximum(ST - K, 0.0)
    else:
        payoff = np.maximum(K - ST, 0.0)

    # 折现取平均
    discount = np.exp(-r * T)
    price = discount * np.mean(payoff)
    std_error = discount * np.std(payoff) / np.sqrt(n_paths)

    return {
        'price': price,
        'std_error': std_error,
        'confidence_interval_95': (price - 1.96 * std_error, price + 1.96 * std_error),
        'n_paths': n_paths,
    }


# ============================================================================
# 2. 对偶变量法（Antithetic Variates）
# ============================================================================

def mc_antithetic(S0, K, T, r, sigma, option_type='call', q=0.0,
                  n_paths=100000, random_state=RANDOM_STATE):
    """
    对偶变量法（Antithetic Variates）—— 降低方差

    原理:
        对于每个随机数 Z，同时使用 Z 和 -Z 生成两条路径。
        由于 Z 和 -Z 完全负相关，两者的平均具有更小的方差。

    方法:
        1. 生成 n_paths/2 个随机数 Z
        2. 计算 S_T1 = S0 * exp(... + sigma*sqrt(T)*Z) 和 S_T2 = S0 * exp(... + sigma*sqrt(T)*(-Z))
        3. 对两条路径的收益取平均: payoff_avg = (payoff1 + payoff2) / 2
        4. 折现取平均

    方差降低效果:
        Var(payoff_avg) = (Var1 + Var2 + 2*Cov) / 4
        由于对偶性，Cov < 0，因此方差降低

    参数: 同 mc_european()
    返回: 同 mc_european()
    """
    n_half = n_paths // 2  # 对偶变量法实际只需要一半的随机数
    rng = np.random.RandomState(random_state)
    Z = rng.standard_normal(n_half)

    # 生成对偶路径
    drift = (r - q - 0.5 * sigma ** 2) * T
    vol = sigma * np.sqrt(T)

    ST1 = S0 * np.exp(drift + vol * Z)       # 正向
    ST2 = S0 * np.exp(drift + vol * (-Z))     # 对偶（使用-Z）

    # 计算收益
    if option_type == 'call':
        payoff1 = np.maximum(ST1 - K, 0.0)
        payoff2 = np.maximum(ST2 - K, 0.0)
    else:
        payoff1 = np.maximum(K - ST1, 0.0)
        payoff2 = np.maximum(K - ST2, 0.0)

    # 对偶平均
    payoff = (payoff1 + payoff2) / 2.0

    discount = np.exp(-r * T)
    price = discount * np.mean(payoff)
    std_error = discount * np.std(payoff) / np.sqrt(n_half)

    return {
        'price': price,
        'std_error': std_error,
        'confidence_interval_95': (price - 1.96 * std_error, price + 1.96 * std_error),
        'n_paths': n_paths,
    }


# ============================================================================
# 3. 控制变量法（Control Variates）
# ============================================================================

def mc_control_variate(S0, K, T, r, sigma, option_type='call', q=0.0,
                       n_paths=100000, random_state=RANDOM_STATE):
    """
    控制变量法（Control Variates）—— 利用已知解析解降低方差

    原理:
        选择一个与期权收益高度相关、且期望已知的控制变量 Y。
        调整后的估计量: V_cv = V_mc - beta * (Y - E[Y])

        其中 beta = Cov(V, Y) / Var(Y) 为最优系数。

    本实现使用 S_T（到期价格）作为控制变量:
        - E[S_T] = S0 * exp((r-q)*T) （已知）
        - S_T 与期权收益高度相关

    参数: 同 mc_european()
    返回:
        dict: 包含以下字段:
            - 'price': 期权价格估计
            - 'std_error': 标准误差
            - 'confidence_interval_95': 95%置信区间
            - 'n_paths': 模拟路径数
            - 'beta': 最优控制系数
    """
    rng = np.random.RandomState(random_state)
    Z = rng.standard_normal(n_paths)

    drift = (r - q - 0.5 * sigma ** 2) * T
    vol = sigma * np.sqrt(T)
    ST = S0 * np.exp(drift + vol * Z)

    # 期权收益
    if option_type == 'call':
        payoff = np.maximum(ST - K, 0.0)
    else:
        payoff = np.maximum(K - ST, 0.0)

    # 控制变量: S_T，已知期望 E[S_T] = S0 * exp((r-q)*T)
    E_ST = S0 * np.exp((r - q) * T)

    # 计算最优系数 beta = Cov(payoff, ST) / Var(ST)
    cov_matrix = np.cov(payoff, ST)
    cov_payoff_ST = cov_matrix[0, 1]
    var_ST = cov_matrix[1, 1]
    beta = cov_payoff_ST / var_ST

    # 调整后的收益
    adjusted_payoff = payoff - beta * (ST - E_ST)

    discount = np.exp(-r * T)
    price = discount * np.mean(adjusted_payoff)
    std_error = discount * np.std(adjusted_payoff) / np.sqrt(n_paths)

    return {
        'price': price,
        'std_error': std_error,
        'confidence_interval_95': (price - 1.96 * std_error, price + 1.96 * std_error),
        'n_paths': n_paths,
        'beta': beta,
    }


# ============================================================================
# 4. Longstaff-Schwartz 最小二乘蒙特卡洛 - 美式期权
# ============================================================================

def mc_american_ls(S0, K, T, r, sigma, option_type='put', q=0.0,
                   n_paths=10000, n_steps=50, random_state=RANDOM_STATE):
    """
    Longstaff-Schwartz 最小二乘蒙特卡洛法 - 美式期权定价

    算法原理（Longstaff & Schwartz, 2001）:
        1. 前向模拟 n_paths 条价格路径
        2. 在到期时，期权价值 = 内在价值
        3. 后向迭代，在每个可提前行权时间点:
           a. 对处于价内（ITM）的路径，使用回归估计继续持有价值（continuation value）
              回归变量: S_t 的多项式基函数 [1, S, S^2, S^3]
              被回归变量: 下一时间步期权价值的折现值
           b. 比较继续持有价值与立即行权价值
           c. 若行权价值 > 继续持有价值，则行权
        4. 将各路径的行权收益折现到时间0，取平均

    参数:
        S0: 标的资产初始价格
        K: 执行价
        T: 到期时间（年）
        r: 无风险利率
        sigma: 波动率
        option_type: 'call' 或 'put'（美式期权通常用put，因为美式call在无股息时等于欧式call）
        q: 连续股息收益率
        n_paths: 模拟路径数
        n_steps: 时间步数（即可提前行权的时间点数）
        random_state: 随机种子

    返回:
        dict: 包含以下字段:
            - 'price': 美式期权价格估计
            - 'n_paths': 模拟路径数
            - 'n_steps': 时间步数
            - 'early_exercise_boundary': 提前行权边界（每个时间点的临界价格）
    """
    # 前向模拟价格路径
    S = simulate_gbm_paths(S0, T, r, sigma, n_paths, n_steps, q, random_state)
    dt = T / n_steps
    discount = np.exp(-r * dt)

    # 内在价值函数
    def intrinsic(s):
        if option_type == 'call':
            return np.maximum(s - K, 0.0)
        else:
            return np.maximum(K - s, 0.0)

    # 期权价值矩阵 V (n_paths, n_steps+1)
    V = np.zeros_like(S)
    V[:, -1] = intrinsic(S[:, -1])  # 到期时价值 = 内在价值

    # 记录提前行权边界
    exercise_boundary = np.zeros(n_steps + 1)
    exercise_boundary[-1] = K  # 到期时边界即执行价

    # 后向迭代: 从 T-dt 到 dt
    for t in range(n_steps - 1, 0, -1):
        # 当前时间步的内在价值
        ex_val = intrinsic(S[:, t])

        # 找出价内（ITM）的路径
        itm = ex_val > 0

        if np.sum(itm) < 4:
            # ITM路径太少，无法回归，直接折现
            V[:, t] = discount * V[:, t + 1]
            continue

        # 回归变量: S_t 的多项式基函数 [1, S, S^2, S^3]
        X = S[itm, t]
        # 构建设计矩阵
        basis = np.column_stack([
            np.ones(len(X)),
            X,
            X ** 2,
            X ** 3
        ])

        # 被回归变量: 下一时间步期权价值的折现值
        Y = discount * V[itm, t + 1]

        # 最小二乘回归
        coeffs, _, _, _ = np.linalg.lstsq(basis, Y, rcond=None)

        # 估计继续持有价值
        cont_val = basis @ coeffs

        # 比较继续持有 vs 立即行权
        exercise = ex_val[itm] > cont_val

        # 更新期权价值
        V_itm = np.zeros(np.sum(itm))
        V_itm[exercise] = ex_val[itm][exercise]          # 行权
        V_itm[~exercise] = discount * V[itm, t + 1][~exercise]  # 继续持有
        V[itm, t] = V_itm

        # 非ITM路径: 继续持有
        V[~itm, t] = discount * V[~itm, t + 1]

        # 记录提前行权边界（ITM路径中行权的最小价格）
        if option_type == 'put':
            # 对put，行权边界是价内路径中行权的最大价格
            exercised_prices = X[exercise]
            if len(exercised_prices) > 0:
                exercise_boundary[t] = np.max(exercised_prices)
            else:
                exercise_boundary[t] = 0.0
        else:
            # 对call，行权边界是价内路径中行权的最小价格
            exercised_prices = X[exercise]
            if len(exercised_prices) > 0:
                exercise_boundary[t] = np.min(exercised_prices)
            else:
                exercise_boundary[t] = np.inf

    # 时间0的价值: 折现到当前
    price = discount * np.mean(V[:, 1])

    return {
        'price': price,
        'n_paths': n_paths,
        'n_steps': n_steps,
        'early_exercise_boundary': exercise_boundary,
    }


# ============================================================================
# 5. 亚式期权（路径依赖期权）
# ============================================================================

def mc_asian(S0, K, T, r, sigma, option_type='call', q=0.0, avg_type='arithmetic',
             n_paths=100000, n_steps=252, random_state=RANDOM_STATE):
    """
    亚式期权定价（蒙特卡洛模拟）

    亚式期权的收益取决于标的资产价格在期权存续期内的平均值。

    算术平均亚式期权:
        收益 = max(A_arith - K, 0) （看涨）或 max(K - A_arith, 0) （看跌）
        其中 A_arith = (1/n) * sum(S_t)

    几何平均亚式期权:
        收益 = max(A_geom - K, 0) （看涨）或 max(K - A_geom, 0) （看跌）
        其中 A_geom = exp((1/n) * sum(ln(S_t)))

    注意: 几何平均亚式期权有解析解（可利用对数正态分布性质），
    可作为算术平均亚式期权的控制变量。

    参数:
        S0: 标的资产初始价格
        K: 执行价
        T: 到期时间（年）
        r: 无风险利率
        sigma: 波动率
        option_type: 'call' 或 'put'
        avg_type: 'arithmetic'（算术平均）或 'geometric'（几何平均）
        n_paths: 模拟路径数
        n_steps: 观测点数（如252表示每日观测）
        random_state: 随机种子

    返回:
        dict: 包含以下字段:
            - 'price': 亚式期权价格
            - 'std_error': 标准误差
            - 'n_paths': 模拟路径数
            - 'avg_type': 平均类型
    """
    # 模拟价格路径
    S = simulate_gbm_paths(S0, T, r, sigma, n_paths, n_steps, q, random_state)

    # 计算平均价格（排除初始价格S0）
    S_obs = S[:, 1:]  # 不包含t=0的价格

    if avg_type == 'arithmetic':
        # 算术平均
        A = np.mean(S_obs, axis=1)
    elif avg_type == 'geometric':
        # 几何平均 = exp(mean(ln(S)))
        A = np.exp(np.mean(np.log(S_obs), axis=1))
    else:
        raise ValueError(f"不支持的平均类型: {avg_type}，请使用 'arithmetic' 或 'geometric'")

    # 计算收益
    if option_type == 'call':
        payoff = np.maximum(A - K, 0.0)
    else:
        payoff = np.maximum(K - A, 0.0)

    # 折现取平均
    discount = np.exp(-r * T)
    price = discount * np.mean(payoff)
    std_error = discount * np.std(payoff) / np.sqrt(n_paths)

    return {
        'price': price,
        'std_error': std_error,
        'n_paths': n_paths,
        'avg_type': avg_type,
    }


# ============================================================================
# 6. 收敛性分析
# ============================================================================

def convergence_analysis(S0, K, T, r, sigma, option_type='call', q=0.0,
                         method='standard', random_state=RANDOM_STATE):
    """
    蒙特卡洛收敛性分析 —— 不同模拟次数下的定价误差

    分析随着模拟路径数增加，蒙特卡洛估计如何收敛到真实值。
    以BS解析解作为基准（仅对欧式期权有意义）。

    参数:
        S0, K, T, r, sigma, option_type, q: 同前
        method: 'standard'（标准MC）、'antithetic'（对偶变量）、'control_variate'（控制变量）
        random_state: 随机种子

    返回:
        dict: 包含以下字段:
            - 'n_paths_list': 模拟路径数列表
            - 'prices': 对应的期权价格估计列表
            - 'std_errors': 对应的标准误差列表
            - 'bs_price': BS解析解（基准）
            - 'errors': 绝对误差列表
    """
    # BS解析解作为基准
    bs = bs_price(S0, K, T, r, sigma, option_type, q)

    # 不同的模拟次数
    n_paths_list = [1000, 5000, 10000, 50000, 100000, 500000]
    prices = []
    std_errors = []

    for n in n_paths_list:
        if method == 'standard':
            result = mc_european(S0, K, T, r, sigma, option_type, q, n, random_state)
        elif method == 'antithetic':
            result = mc_antithetic(S0, K, T, r, sigma, option_type, q, n, random_state)
        elif method == 'control_variate':
            result = mc_control_variate(S0, K, T, r, sigma, option_type, q, n, random_state)
        else:
            raise ValueError(f"不支持的方法: {method}")

        prices.append(result['price'])
        std_errors.append(result['std_error'])

    prices = np.array(prices)
    std_errors = np.array(std_errors)
    errors = np.abs(prices - bs)

    return {
        'n_paths_list': n_paths_list,
        'prices': prices,
        'std_errors': std_errors,
        'bs_price': bs,
        'errors': errors,
    }
