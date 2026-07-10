"""
有限差分法定价模块（Finite Difference Method）
=================================================

本模块实现以下功能：
1. 显式差分法（Explicit FDM）
2. 隐式差分法（Implicit FDM）
3. Crank-Nicolson 差分法（theta=0.5的隐式-显式混合）
4. 欧式期权和美式期权定价
5. 收敛性分析

Black-Scholes PDE:
-------------------
    dV/dt + (r-q)*S*dV/dS + 0.5*sigma^2*S^2*d^2V/dS^2 - r*V = 0

时间反向化（令 tau = T - t，到期时 tau=0）:
    dV/dtau = (r-q)*S*dV/dS + 0.5*sigma^2*S^2*d^2V/dS^2 - r*V

网格构建:
---------
    S方向: [0, S_max]，N+1个点，dS = S_max/N
    tau方向: [0, T]，M+1个点，dt = T/M
    V_i^n 表示 S=S_i=i*dS, tau=n*dt 处的期权价值

空间离散化:
    dV/dS ≈ (V_{i+1} - V_{i-1}) / (2*dS)
    d^2V/dS^2 ≈ (V_{i+1} - 2*V_i + V_{i-1}) / dS^2

代入PDE后，系数（不含dt）:
    a_i = 0.5 * (sigma^2 * i^2 - (r-q) * i)
    b_i = -(sigma^2 * i^2 + r)
    c_i = 0.5 * (sigma^2 * i^2 + (r-q) * i)

显式差分:
    V_i^{n+1} = dt*a_i * V_{i-1}^n + (1 + dt*b_i) * V_i^n + dt*c_i * V_{i+1}^n

隐式差分:
    -dt*a_i * V_{i-1}^{n+1} + (1 - dt*b_i) * V_i^{n+1} - dt*c_i * V_{i+1}^{n+1} = V_i^n

Crank-Nicolson (theta=0.5):
    -0.5*dt*a_i * V_{i-1}^{n+1} + (1 - 0.5*dt*b_i) * V_i^{n+1} - 0.5*dt*c_i * V_{i+1}^{n+1}
    = 0.5*dt*a_i * V_{i-1}^n + (1 + 0.5*dt*b_i) * V_i^n + 0.5*dt*c_i * V_{i+1}^n

边界条件:
    S=0:   看跌 V = K*exp(-r*tau)，看涨 V = 0
    S=S_max: 看跌 V ≈ 0，看涨 V ≈ S_max*exp(-q*tau) - K*exp(-r*tau)

美式期权:
    在每个时间步后施加提前行权约束: V = max(V, intrinsic_value)
"""

import numpy as np
from scipy.linalg import solve_banded

from .black_scholes import bs_price


# ============================================================================
# 辅助函数：构建网格和边界条件
# ============================================================================

def _build_grid(S0, K, T, r, sigma, q, N, M, S_max_factor=3.0):
    """
    构建有限差分网格

    参数:
        S0: 标的资产价格（用于确定S_max）
        K: 执行价
        T: 到期时间
        r, sigma, q: 利率、波动率、股息率
        N: 价格方向网格数
        M: 时间方向网格数
        S_max_factor: S_max = max(S0, K) * factor

    返回:
        dict: 包含网格信息
    """
    S_max = max(S0, K) * S_max_factor
    dS = S_max / N
    dt = T / M

    S_grid = np.linspace(0, S_max, N + 1)
    tau_grid = np.linspace(0, T, M + 1)
    i_grid = np.arange(N + 1)  # i = 0, 1, ..., N

    # 空间离散化系数（不含dt）
    a = 0.5 * (sigma ** 2 * i_grid ** 2 - (r - q) * i_grid)
    b = -(sigma ** 2 * i_grid ** 2 + r)
    c = 0.5 * (sigma ** 2 * i_grid ** 2 + (r - q) * i_grid)

    return {
        'S_max': S_max, 'dS': dS, 'dt': dt,
        'S_grid': S_grid, 'tau_grid': tau_grid,
        'i_grid': i_grid, 'a': a, 'b': b, 'c': c,
        'N': N, 'M': M
    }


def _payoff(S, K, option_type):
    """计算内在价值（收益函数）"""
    if option_type == 'call':
        return np.maximum(S - K, 0.0)
    else:
        return np.maximum(K - S, 0.0)


def _boundary_conditions(tau, K, r, q, option_type, S_max):
    """
    计算边界条件

    S=0边界:
        看跌: V = K * exp(-r*tau)  （到期时确定收到K）
        看涨: V = 0  （标的价格为0时看涨期权无价值）

    S=S_max边界:
        看跌: V ≈ 0  （深度价外，价值几乎为0）
        看涨: V ≈ S_max*exp(-q*tau) - K*exp(-r*tau)  （深度价内）

    参数:
        tau: 当前时间（tau=0为到期，tau=T为当前）
        K, r, q: 执行价、利率、股息率
        option_type: 'call' 或 'put'
        S_max: 网格上界

    返回:
        tuple: (V_at_0, V_at_S_max)
    """
    if option_type == 'call':
        V_0 = 0.0
        V_Smax = S_max * np.exp(-q * tau) - K * np.exp(-r * tau)
        V_Smax = max(V_Smax, 0.0)  # 期权价值非负
    else:  # put
        V_0 = K * np.exp(-r * tau)
        V_Smax = 0.0

    return V_0, V_Smax


# ============================================================================
# 1. 显式差分法
# ============================================================================

def fdm_explicit(S0, K, T, r, sigma, option_type='put', q=0.0,
                 american=True, N=200, M=200, S_max_factor=3.0):
    """
    显式差分法（Explicit Finite Difference Method）

    显式差分直接从已知时间步计算下一时间步:
        V_i^{n+1} = dt*a_i * V_{i-1}^n + (1 + dt*b_i) * V_i^n + dt*c_i * V_{i+1}^n

    稳定性条件: dt <= dS^2 / (sigma^2 * S_max^2)
    若不满足，可能产生数值不稳定。

    参数:
        S0: 标的资产价格
        K: 执行价
        T: 到期时间（年）
        r: 无风险利率
        sigma: 波动率
        option_type: 'call' 或 'put'
        q: 连续股息收益率
        american: True为美式期权，False为欧式
        N: 价格方向网格点数
        M: 时间方向网格点数
        S_max_factor: S_max = max(S0, K) * factor

    返回:
        dict: 包含期权价格和网格信息
    """
    grid = _build_grid(S0, K, T, r, sigma, q, N, M, S_max_factor)
    dS = grid['dS']
    dt = grid['dt']
    S_grid = grid['S_grid']
    a = grid['a']
    b = grid['b']
    c = grid['c']

    # 稳定性检查
    stability_limit = dS ** 2 / (sigma ** 2 * grid['S_max'] ** 2)
    if dt > stability_limit:
        # 自动调整M以满足稳定性
        M_needed = int(np.ceil(T / stability_limit)) + 1
        grid = _build_grid(S0, K, T, r, sigma, q, N, M_needed, S_max_factor)
        dt = grid['dt']
        M = M_needed

    # 初始化: 到期时 tau=0，V = 收益
    V = _payoff(S_grid, K, option_type)

    # 时间步进（从tau=0向tau=T，即从到期向当前）
    for n in range(M):
        V_new = np.zeros(N + 1)

        # 内部点: 显式差分公式
        for i in range(1, N):
            V_new[i] = (dt * a[i] * V[i - 1] +
                        (1 + dt * b[i]) * V[i] +
                        dt * c[i] * V[i + 1])

        # 边界条件
        tau = (n + 1) * dt
        V_0, V_Smax = _boundary_conditions(tau, K, r, q, option_type, grid['S_max'])
        V_new[0] = V_0
        V_new[N] = V_Smax

        # 美式期权: 施加提前行权约束
        if american:
            V_new = np.maximum(V_new, _payoff(S_grid, K, option_type))

        V = V_new

    # 插值得到S0处的期权价格
    price = np.interp(S0, S_grid, V)

    return {
        'price': price,
        'method': 'explicit',
        'N': N,
        'M': M,
        'american': american,
        'S_grid': S_grid,
        'V_grid': V,
    }


# ============================================================================
# 2. 隐式差分法
# ============================================================================

def fdm_implicit(S0, K, T, r, sigma, option_type='put', q=0.0,
                 american=True, N=200, M=200, S_max_factor=3.0):
    """
    隐式差分法（Implicit Finite Difference Method）

    隐式差分需要求解三对角线性方程组:
        -dt*a_i * V_{i-1}^{n+1} + (1 - dt*b_i) * V_i^{n+1} - dt*c_i * V_{i+1}^{n+1} = V_i^n

    即: A * V^{n+1} = V^n + boundary

    隐式差分无条件稳定（不受dt/dS比限制），但每步需要求解线性系统。

    参数: 同 fdm_explicit()
    返回: 同 fdm_explicit()
    """
    grid = _build_grid(S0, K, T, r, sigma, q, N, M, S_max_factor)
    dt = grid['dt']
    S_grid = grid['S_grid']
    a = grid['a']
    b = grid['b']
    c = grid['c']
    N = grid['N']
    M = grid['M']

    # 构建三对角矩阵 A（带状存储格式）
    # A * V^{n+1} = V^n + rhs_boundary
    # 对角线: 1 - dt*b_i (下、主、上)
    # 下对角线: -dt*a_i
    # 上对角线: -dt*c_i
    #
    # solve_banded 格式: ab[u+i-j, j] = A[i, j]
    # 对于三对角: ab[0, :] = 上对角线, ab[1, :] = 主对角线, ab[2, :] = 下对角线

    # 主对角线 (i=1..N-1)
    main_diag = 1 - dt * b[1:N]
    # 下对角线 (对应a, i=2..N-1, 消去V_{i-1})
    lower_diag = -dt * a[2:N]
    # 上对角线 (对应c, i=1..N-2, 消去V_{i+1})
    upper_diag = -dt * c[1:N - 1]

    # 带状矩阵 (solve_banded 格式: 3行)
    ab = np.zeros((3, N - 1))
    ab[0, 1:] = upper_diag      # 上对角线 (偏移+1)
    ab[1, :] = main_diag        # 主对角线
    ab[2, :-1] = lower_diag     # 下对角线 (偏移-1)

    # 初始化: 到期时 V = 收益
    V = _payoff(S_grid, K, option_type)

    # 时间步进
    for n in range(M):
        # 右端向量: V^n 的内部点
        rhs = V[1:N].copy()

        # 处理边界条件
        tau = (n + 1) * dt
        V_0, V_Smax = _boundary_conditions(tau, K, r, q, option_type, grid['S_max'])

        # 边界对右端向量的贡献
        rhs[0] += dt * a[1] * V_0       # i=1, 下边界项
        rhs[-1] += dt * c[N - 1] * V_Smax  # i=N-1, 上边界项

        # 求解三对角系统
        V_inner = solve_banded((1, 1), ab, rhs)

        # 组装完整解
        V_new = np.zeros(N + 1)
        V_new[0] = V_0
        V_new[1:N] = V_inner
        V_new[N] = V_Smax

        # 美式期权: 施加提前行权约束
        if american:
            V_new = np.maximum(V_new, _payoff(S_grid, K, option_type))

        V = V_new

    # 插值得到S0处的期权价格
    price = np.interp(S0, S_grid, V)

    return {
        'price': price,
        'method': 'implicit',
        'N': N,
        'M': M,
        'american': american,
        'S_grid': S_grid,
        'V_grid': V,
    }


# ============================================================================
# 3. Crank-Nicolson 差分法
# ============================================================================

def fdm_crank_nicolson(S0, K, T, r, sigma, option_type='put', q=0.0,
                        american=True, N=200, M=200, S_max_factor=3.0):
    """
    Crank-Nicolson 差分法

    Crank-Nicolson 是显式和隐式的平均（theta=0.5），具有二阶时间精度:
        左端: -0.5*dt*a_i * V_{i-1}^{n+1} + (1 - 0.5*dt*b_i) * V_i^{n+1} - 0.5*dt*c_i * V_{i+1}^{n+1}
        右端: 0.5*dt*a_i * V_{i-1}^n + (1 + 0.5*dt*b_i) * V_i^n + 0.5*dt*c_i * V_{i+1}^n

    即: A * V^{n+1} = B * V^n + boundary

    无条件稳定且二阶精度，是最常用的有限差分方法。

    参数: 同 fdm_explicit()
    返回: 同 fdm_explicit()
    """
    grid = _build_grid(S0, K, T, r, sigma, q, N, M, S_max_factor)
    dt = grid['dt']
    S_grid = grid['S_grid']
    a = grid['a']
    b = grid['b']
    c = grid['c']
    N = grid['N']
    M = grid['M']

    # 构建左端矩阵 A (隐式部分, theta=0.5)
    # A: 主对角线 1 - 0.5*dt*b_i
    #    下对角线 -0.5*dt*a_i
    #    上对角线 -0.5*dt*c_i
    main_diag_A = 1 - 0.5 * dt * b[1:N]
    lower_diag_A = -0.5 * dt * a[2:N]
    upper_diag_A = -0.5 * dt * c[1:N - 1]

    ab_A = np.zeros((3, N - 1))
    ab_A[0, 1:] = upper_diag_A
    ab_A[1, :] = main_diag_A
    ab_A[2, :-1] = lower_diag_A

    # 右端矩阵 B (显式部分, theta=0.5)
    # B: 主对角线 1 + 0.5*dt*b_i
    #    下对角线 0.5*dt*a_i
    #    上对角线 0.5*dt*c_i
    main_diag_B = 1 + 0.5 * dt * b[1:N]
    lower_diag_B = 0.5 * dt * a[2:N]
    upper_diag_B = 0.5 * dt * c[1:N - 1]

    # 初始化: 到期时 V = 收益
    V = _payoff(S_grid, K, option_type)

    # 时间步进
    for n in range(M):
        # 计算 B * V^n (右端显式部分)
        rhs = np.zeros(N - 1)
        for i in range(N - 1):
            idx = i + 1  # 实际的 i 索引 (1 到 N-1)
            val = main_diag_B[i] * V[idx]
            if idx > 1:
                val += lower_diag_B[i - 1] * V[idx - 1]
            if idx < N - 1:
                val += upper_diag_B[i] * V[idx + 1]
            rhs[i] = val

        # 边界条件
        tau = (n + 1) * dt
        V_0, V_Smax = _boundary_conditions(tau, K, r, q, option_type, grid['S_max'])

        # 边界对右端向量的贡献
        rhs[0] += 0.5 * dt * a[1] * V_0 + 0.5 * dt * a[1] * V[0]
        rhs[-1] += 0.5 * dt * c[N - 1] * V_Smax + 0.5 * dt * c[N - 1] * V[N]

        # 求解 A * V^{n+1} = rhs
        V_inner = solve_banded((1, 1), ab_A, rhs)

        # 组装完整解
        V_new = np.zeros(N + 1)
        V_new[0] = V_0
        V_new[1:N] = V_inner
        V_new[N] = V_Smax

        # 美式期权: 施加提前行权约束
        if american:
            V_new = np.maximum(V_new, _payoff(S_grid, K, option_type))

        V = V_new

    # 插值得到S0处的期权价格
    price = np.interp(S0, S_grid, V)

    return {
        'price': price,
        'method': 'crank_nicolson',
        'N': N,
        'M': M,
        'american': american,
        'S_grid': S_grid,
        'V_grid': V,
    }


# ============================================================================
# 统一接口
# ============================================================================

def fdm_european(S0, K, T, r, sigma, option_type='put', q=0.0,
                 method='crank_nicolson', N=200, M=200, S_max_factor=3.0):
    """
    有限差分法 - 欧式期权定价

    参数:
        method: 'explicit', 'implicit', 'crank_nicolson'
        其他参数同前

    返回: 同 fdm_explicit()
    """
    if method == 'explicit':
        return fdm_explicit(S0, K, T, r, sigma, option_type, q,
                            american=False, N=N, M=M, S_max_factor=S_max_factor)
    elif method == 'implicit':
        return fdm_implicit(S0, K, T, r, sigma, option_type, q,
                            american=False, N=N, M=M, S_max_factor=S_max_factor)
    elif method == 'crank_nicolson':
        return fdm_crank_nicolson(S0, K, T, r, sigma, option_type, q,
                                  american=False, N=N, M=M, S_max_factor=S_max_factor)
    else:
        raise ValueError(f"不支持的差分方法: {method}")


def fdm_american(S0, K, T, r, sigma, option_type='put', q=0.0,
                 method='crank_nicolson', N=200, M=200, S_max_factor=3.0):
    """
    有限差分法 - 美式期权定价

    参数: 同 fdm_european()
    返回: 同 fdm_explicit()
    """
    if method == 'explicit':
        return fdm_explicit(S0, K, T, r, sigma, option_type, q,
                            american=True, N=N, M=M, S_max_factor=S_max_factor)
    elif method == 'implicit':
        return fdm_implicit(S0, K, T, r, sigma, option_type, q,
                            american=True, N=N, M=M, S_max_factor=S_max_factor)
    elif method == 'crank_nicolson':
        return fdm_crank_nicolson(S0, K, T, r, sigma, option_type, q,
                                  american=True, N=N, M=M, S_max_factor=S_max_factor)
    else:
        raise ValueError(f"不支持的差分方法: {method}")


# ============================================================================
# 收敛性分析
# ============================================================================

def fdm_convergence_analysis(S0, K, T, r, sigma, option_type='put', q=0.0,
                              method='crank_nicolson', american=True,
                              N_list=None, M_factor=2):
    """
    有限差分法收敛性分析 —— 不同网格密度下的定价误差

    以BS解析解作为基准（仅对欧式期权），分析随网格密度增加的收敛情况。

    参数:
        N_list: 网格点数列表（默认 [50, 100, 200, 400, 800]）
        M_factor: M = N * M_factor（时间步数为空间步数的倍数）
        其他参数同前

    返回:
        dict: 包含网格信息、价格和误差
    """
    if N_list is None:
        N_list = [50, 100, 200, 400, 800]

    prices = []
    for N in N_list:
        M = N * M_factor
        if american:
            result = fdm_american(S0, K, T, r, sigma, option_type, q, method, N, M)
        else:
            result = fdm_european(S0, K, T, r, sigma, option_type, q, method, N, M)
        prices.append(result['price'])

    prices = np.array(prices)

    # 基准价格
    if not american:
        # 欧式期权有BS解析解
        bs = bs_price(S0, K, T, r, sigma, option_type, q)
        errors = np.abs(prices - bs)
    else:
        # 美式期权没有解析解，以最细网格作为参考
        bs = prices[-1]
        errors = np.abs(prices - bs)

    return {
        'N_list': N_list,
        'M_list': [n * M_factor for n in N_list],
        'prices': prices,
        'reference': bs,
        'errors': errors,
    }
