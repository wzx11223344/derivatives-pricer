"""
COS方法 (Fourier Cosine Expansion Method)
==========================================

基于 Fang & Oosterlee (2008) 论文实现的 COS 方法，通过特征函数的
Fourier 余弦展开对欧式期权进行高效定价。

核心思想：
    将期权价格的积分表示中的概率密度函数用 Fourier 余弦级数近似，
    然后将密度函数的余弦系数与特征函数关联，从而避免数值积分。

数学公式：
    V ≈ e^{-rT} * (2/(b-a)) * Σ' Re[φ(ω_k) * e^{-iω_k*a}] * V_k

    其中：
    - ω_k = k*π/(b-a)
    - φ(·) 为 X_T = ln(S_T/K) 的特征函数
    - V_k 为期权收益函数的余弦系数
    - Σ' 表示第一项乘以 1/2

支持分布：
    - 正态分布 (GBM, 对应 Black-Scholes)
    - Variance-Gamma (VG) 分布
    - Normal Inverse Gaussian (NIG) 分布

参考文献：
    Fang, F., & Oosterlee, C. W. (2008). A novel pricing method for European
    options based on the Fourier-cosine series expansion. SIAM Journal on
    Scientific Computing, 31(2), 826-848.
"""

import numpy as np
from typing import Optional, Dict, Tuple


# =============================================================================
# 特征函数定义
# =============================================================================

def characteristic_function_normal(
    u: np.ndarray,
    S0: float,
    K: float,
    r: float,
    q: float,
    sigma: float,
    T: float
) -> np.ndarray:
    """
    正态分布（GBM）的特征函数

    X_T = ln(S_T/K) 在风险中性测度下服从：
    X_T = ln(S0/K) + (r - q - σ²/2)T + σW_T

    特征函数：
    φ(u) = exp(iu[x + μT] - σ²u²T/2)
    其中 x = ln(S0/K), μ = r - q - σ²/2

    参数：
        u: 频率向量 (numpy数组)
        S0: 标的资产现价
        K: 行权价
        r: 无风险利率
        q: 连续股息收益率
        sigma: 波动率
        T: 到期时间

    返回：
        特征函数值 (复数数组)
    """
    x = np.log(S0 / K)  # 对数货币性
    mu = r - q - 0.5 * sigma ** 2  # 风险中性漂移
    # φ(u) = exp(iu(x + μT) - σ²u²T/2)
    phi = np.exp(1j * u * (x + mu * T) - 0.5 * sigma ** 2 * u ** 2 * T)
    return phi


def characteristic_function_vg(
    u: np.ndarray,
    S0: float,
    K: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    theta: float,
    nu: float
) -> np.ndarray:
    """
    Variance-Gamma (VG) 分布的特征函数

    VG 过程是布朗运动的时间变换，其中时间变换为 Gamma 过程。
    风险中性 VG 模型下：
    X_T = ln(S0/K) + (r - q + ω)T + θ*G_T + σ*√(G_T)*Z

    其中 G_T ~ Gamma(T/ν, ν) 为 Gamma 过程，
    ω = (1/ν)*ln(1 - θν - σ²ν/2) 为鞅修正项。

    特征函数：
    φ(u) = exp(iu[x + (r-q+ω)T]) * (1 - iθνu + σ²νu²/2)^{-T/ν}

    参数：
        u: 频率向量
        S0, K, r, q, sigma, T: 标准参数
        theta: VG 漂移参数
        nu: VG 方差率参数 (variance rate)

    返回：
        特征函数值 (复数数组)
    """
    x = np.log(S0 / K)
    # 鞅修正项，确保折现资产价格为鞅
    omega = (1.0 / nu) * np.log(1.0 - theta * nu - 0.5 * sigma ** 2 * nu)
    # VG 特征函数
    # (1 - iθνu + σ²νu²/2)^{-T/ν}
    base = 1.0 - 1j * theta * nu * u + 0.5 * sigma ** 2 * nu * u ** 2
    phi = np.exp(1j * u * (x + (r - q + omega) * T)) * base ** (-T / nu)
    return phi


def characteristic_function_nig(
    u: np.ndarray,
    S0: float,
    K: float,
    r: float,
    q: float,
    alpha: float,
    beta: float,
    delta: float,
    T: float
) -> np.ndarray:
    """
    Normal Inverse Gaussian (NIG) 分布的特征函数

    NIG 过程是逆高斯分布与布朗运动的混合。
    风险中性 NIG 模型下，特征函数为：

    φ(u) = exp(iu[x + (r-q+ω)T] + Tδ(√(α²-β²) - √(α²-(β+iu)²)))

    其中 ω = δ(√(α²-β²) - √(α²-(β+1)²)) 为鞅修正项。

    约束条件：α > 0, |β| < α, δ > 0

    参数：
        u: 频率向量
        S0, K, r, q, T: 标准参数
        alpha: NIG 参数 α (>0)
        beta: NIG 参数 β (|β|<α)
        delta: NIG 参数 δ (>0)

    返回：
        特征函数值 (复数数组)
    """
    x = np.log(S0 / K)
    # 鞅修正项
    omega = delta * (
        np.sqrt(alpha ** 2 - beta ** 2) -
        np.sqrt(alpha ** 2 - (beta + 1) ** 2)
    )
    # NIG 特征函数的核心部分
    sqrt_term = np.sqrt(alpha ** 2 - (beta + 1j * u) ** 2)
    phi = np.exp(
        1j * u * (x + (r - q + omega) * T) +
        T * delta * (np.sqrt(alpha ** 2 - beta ** 2) - sqrt_term)
    )
    return phi


# =============================================================================
# 截断区间计算
# =============================================================================

def _compute_truncation_range(
    cf_type: str,
    S0: float,
    K: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    L: float = 10.0,
    cf_params: Optional[Dict] = None
) -> Tuple[float, float]:
    """
    计算积分截断区间 [a, b]

    基于 X_T = ln(S_T/K) 的前两阶累积量：
    c1 = E[X_T] (一阶累积量，即均值)
    c2 = Var[X_T] (二阶累积量，即方差)

    截断区间：[a, b] = [c1 - L*√c2, c1 + L*√√c2]

    参数：
        cf_type: 特征函数类型 ('normal', 'vg', 'nig')
        L: 截断倍数 (默认10，标准选择)
        cf_params: 额外参数字典 (VG: theta, nu; NIG: alpha, beta, delta)

    返回：
        (a, b) 截断区间
    """
    if cf_params is None:
        cf_params = {}

    x = np.log(S0 / K)

    if cf_type == 'normal':
        # GBM: c1 = x + (r-q-σ²/2)T, c2 = σ²T
        c1 = x + (r - q - 0.5 * sigma ** 2) * T
        c2 = sigma ** 2 * T

    elif cf_type == 'vg':
        # VG: 需要计算鞅修正项
        theta = cf_params.get('theta', 0.0)
        nu = cf_params.get('nu', 0.1)
        omega = (1.0 / nu) * np.log(1.0 - theta * nu - 0.5 * sigma ** 2 * nu)
        c1 = x + (r - q + omega) * T + theta * T
        # VG 方差: (σ² + νθ²)T
        c2 = (sigma ** 2 + nu * theta ** 2) * T

    elif cf_type == 'nig':
        # NIG 分布
        alpha = cf_params.get('alpha', 15.0)
        beta = cf_params.get('beta', -5.0)
        delta = cf_params.get('delta', 0.2)
        omega = delta * (
            np.sqrt(alpha ** 2 - beta ** 2) -
            np.sqrt(alpha ** 2 - (beta + 1) ** 2)
        )
        gamma_sq = alpha ** 2 - beta ** 2  # γ² = α² - β²
        c1 = x + (r - q + omega) * T + delta * beta * T / np.sqrt(gamma_sq)
        # NIG 方差: δTα²/(α²-β²)^{3/2}
        c2 = delta * T * alpha ** 2 / gamma_sq ** 1.5

    else:
        raise ValueError(f"不支持的特征函数类型: {cf_type}，可选: 'normal', 'vg', 'nig'")

    # 截断区间
    a = c1 - L * np.sqrt(c2)
    b = c1 + L * np.sqrt(c2)
    return a, b


# =============================================================================
# 期权收益的余弦系数
# =============================================================================

def _call_payoff_coefficients(a: float, b: float, K: float, N: int) -> np.ndarray:
    """
    计算看涨期权收益函数的余弦系数 V_k

    对于看涨期权，收益函数为 v(y) = K*(e^y - 1)^+
    (其中 y = ln(S_T/K)，当 S_T > K 即 y > 0 时有收益)

    V_k = K * (χ_k - ψ_k)

    其中：
    χ_k = ∫₀ᵇ eʸ cos(ω_k(y-a)) dy
        = 1/(1+ω_k²) * [(-1)^k eᵇ - cos(ω_k a) + ω_k sin(ω_k a)]

    ψ_k = ∫₀ᵇ cos(ω_k(y-a)) dy = sin(ω_k a)/ω_k  (k>0)
    ψ_0 = b

    参数：
        a, b: 截断区间
        K: 行权价
        N: 余弦项数

    返回：
        V_k 数组，长度为 N
    """
    k = np.arange(N, dtype=float)
    omega = k * np.pi / (b - a)  # ω_k = kπ/(b-a)

    # 计算 χ_k
    chi = np.zeros(N)
    psi = np.zeros(N)

    # k=0 的特殊情况 (ω_0 = 0)
    # χ_0 = eᵇ - 1 (因为 ∫₀ᵇ eʸ dy = eᵇ - 1)
    chi[0] = np.exp(b) - 1.0
    # ψ_0 = b (∫₀ᵇ 1 dy = b)
    psi[0] = b

    # k > 0 的情况
    k_pos = np.arange(1, N, dtype=float)
    omega_pos = k_pos * np.pi / (b - a)

    # χ_k = 1/(1+ω²) * [(-1)^k eᵇ - cos(ωa) + ω sin(ωa)]
    chi[1:] = (1.0 / (1.0 + omega_pos ** 2)) * (
        (-1.0) ** k_pos * np.exp(b) -
        np.cos(omega_pos * a) +
        omega_pos * np.sin(omega_pos * a)
    )

    # ψ_k = sin(ωa)/ω
    psi[1:] = np.sin(omega_pos * a) / omega_pos

    # V_k = K * (χ_k - ψ_k)
    V_k = K * (chi - psi)
    return V_k


def _put_payoff_coefficients(a: float, b: float, K: float, N: int) -> np.ndarray:
    """
    计算看跌期权收益函数的余弦系数 V_k

    对于看跌期权，收益函数为 v(y) = K*(1 - e^y)^+
    (当 S_T < K 即 y < 0 时有收益)

    V_k = K * (ψ_k' - χ_k')

    其中：
    ψ_k' = ∫ₐ⁰ cos(ω_k(y-a)) dy = -sin(ω_k a)/ω_k  (k>0)
    ψ_0' = -a

    χ_k' = ∫ₐ⁰ eʸ cos(ω_k(y-a)) dy
         = 1/(1+ω_k²) * [cos(ω_k a) - ω_k sin(ω_k a) - eᵃ]  (k>0)
    χ_0' = 1 - eᵃ

    参数：
        a, b: 截断区间
        K: 行权价
        N: 余弦项数

    返回：
        V_k 数组，长度为 N
    """
    k = np.arange(N, dtype=float)
    omega = k * np.pi / (b - a)

    chi_put = np.zeros(N)
    psi_put = np.zeros(N)

    # k=0 的特殊情况
    # χ_0' = ∫ₐ⁰ eʸ dy = 1 - eᵃ
    chi_put[0] = 1.0 - np.exp(a)
    # ψ_0' = ∫ₐ⁰ 1 dy = -a
    psi_put[0] = -a

    # k > 0 的情况
    k_pos = np.arange(1, N, dtype=float)
    omega_pos = k_pos * np.pi / (b - a)

    # χ_k' = 1/(1+ω²) * [cos(ωa) - ω sin(ωa) - eᵃ]
    chi_put[1:] = (1.0 / (1.0 + omega_pos ** 2)) * (
        np.cos(omega_pos * a) -
        omega_pos * np.sin(omega_pos * a) -
        np.exp(a)
    )

    # ψ_k' = -sin(ωa)/ω
    psi_put[1:] = -np.sin(omega_pos * a) / omega_pos

    # V_k = K * (ψ_k' - χ_k')
    V_k = K * (psi_put - chi_put)
    return V_k


# =============================================================================
# COS 方法主定价函数
# =============================================================================

def cos_european(
    S0: float,
    K: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    option_type: str = 'call',
    N: int = 200,
    L: float = 10.0,
    cf_type: str = 'normal',
    cf_params: Optional[Dict] = None
) -> float:
    """
    COS 方法对欧式期权定价

    核心公式：
    V ≈ e^{-rT} * (2/(b-a)) * Σ' Re[φ(ω_k) * e^{-iω_k a}] * V_k

    其中 Σ' 表示第一项乘以 1/2。

    参数：
        S0: 标的资产现价
        K: 行权价
        r: 无风险利率
        q: 连续股息收益率
        sigma: 波动率 (GBM/VG 模型使用)
        T: 到期时间 (年)
        option_type: 'call' 或 'put'
        N: 余弦展开项数 (默认200，越多越精确)
        L: 截断倍数 (默认10)
        cf_type: 特征函数类型
            'normal' - GBM (对应BS模型)
            'vg'     - Variance-Gamma
            'nig'    - Normal Inverse Gaussian
        cf_params: 额外参数
            VG: {'theta': ..., 'nu': ...}
            NIG: {'alpha': ..., 'beta': ..., 'delta': ...}

    返回：
        期权价格
    """
    if cf_params is None:
        cf_params = {}

    # 1. 计算截断区间 [a, b]
    a, b = _compute_truncation_range(cf_type, S0, K, r, q, sigma, T, L, cf_params)

    # 2. 计算频率向量 ω_k
    k = np.arange(N, dtype=float)
    omega = k * np.pi / (b - a)

    # 3. 计算特征函数 φ(ω_k)
    if cf_type == 'normal':
        phi_vals = characteristic_function_normal(u=omega, S0=S0, K=K,
                                                   r=r, q=q, sigma=sigma, T=T)
    elif cf_type == 'vg':
        theta = cf_params.get('theta', 0.0)
        nu = cf_params.get('nu', 0.1)
        phi_vals = characteristic_function_vg(u=omega, S0=S0, K=K, r=r, q=q,
                                               sigma=sigma, T=T, theta=theta, nu=nu)
    elif cf_type == 'nig':
        alpha = cf_params.get('alpha', 15.0)
        beta = cf_params.get('beta', -5.0)
        delta = cf_params.get('delta', 0.2)
        phi_vals = characteristic_function_nig(u=omega, S0=S0, K=K, r=r, q=q,
                                                alpha=alpha, beta=beta,
                                                delta=delta, T=T)
    else:
        raise ValueError(f"不支持的特征函数类型: {cf_type}")

    # 4. 计算 Re[φ(ω_k) * e^{-iω_k a}]
    #    = Re[φ(ω_k)] * cos(ω_k a) + Im[φ(ω_k)] * sin(ω_k a)
    #    等价于 Re[φ(ω_k) * (cos(ω_k a) - i sin(ω_k a))]
    term = np.real(phi_vals * np.exp(-1j * omega * a))

    # 5. 计算收益系数 V_k
    if option_type.lower() == 'call':
        V_k = _call_payoff_coefficients(a, b, K, N)
    elif option_type.lower() == 'put':
        V_k = _put_payoff_coefficients(a, b, K, N)
    else:
        raise ValueError(f"不支持的期权类型: {option_type}，请使用 'call' 或 'put'")

    # 6. 应用带 prime 的求和 (第一项乘以 1/2)
    term[0] *= 0.5

    # 7. 计算期权价格
    #    V = e^{-rT} * (2/(b-a)) * Σ' Re[...] * V_k
    price = np.exp(-r * T) * (2.0 / (b - a)) * np.sum(term * V_k)

    # 8. 数值保护：价格不能为负
    price = max(price, 0.0)

    return float(price)


def cos_price(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option_type: str = 'call',
    q: float = 0.0,
    N: int = 200,
    L: float = 10.0,
    cf_type: str = 'normal',
    cf_params: Optional[Dict] = None
) -> float:
    """
    COS 方法定价的简化接口

    参数：
        S0: 标的资产现价
        K: 行权价
        r: 无风险利率
        sigma: 波动率
        T: 到期时间
        option_type: 'call' 或 'put'
        q: 连续股息收益率 (默认0)
        N: 余弦展开项数 (默认200)
        L: 截断倍数 (默认10)
        cf_type: 特征函数类型 (默认 'normal' 即 GBM)
        cf_params: 额外参数字典

    返回：
        期权价格
    """
    return cos_european(S0, K, r, q, sigma, T, option_type, N, L, cf_type, cf_params)


def cos_accuracy_analysis(
    S0: float,
    K: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    option_type: str = 'call',
    N_list: list = None,
    L: float = 10.0
) -> list:
    """
    COS 方法精度分析：比较不同余弦项数 N 下的定价精度

    使用 Black-Scholes 解析解作为基准，计算 COS 方法在不同 N 下的误差。

    参数：
        S0, K, r, q, sigma, T: 标准期权参数
        option_type: 期权类型
        N_list: 余弦项数列表 (默认 [32, 64, 128, 256, 512])
        L: 截断倍数

    返回：
        列表，每个元素为 (N, cos_price, bs_price, abs_error) 元组
    """
    from .black_scholes import bs_price

    if N_list is None:
        N_list = [32, 64, 128, 256, 512, 1024]

    results = []
    bs = bs_price(S0, K, r, q, sigma, T, option_type)

    for N in N_list:
        cp = cos_european(S0, K, r, q, sigma, T, option_type, N, L, 'normal')
        err = abs(cp - bs)
        results.append((N, cp, bs, err))

    return results
