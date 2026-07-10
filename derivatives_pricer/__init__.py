"""
衍生品定价引擎 (Derivatives Pricer)
====================================

专业级衍生品定价工具包，实现多种定价方法：
- Black-Scholes 解析解（含连续股息收益率q的广义BS模型）
- 蒙特卡洛模拟（标准/对偶变量/控制变量/Longstaff-Schwartz）
- 有限差分法（显式/隐式/Crank-Nicolson）
- COS方法（Fourier余弦展开，Fang & Oosterlee 2008）
- 完整希腊字母计算（Delta/Gamma/Theta/Vega/Rho）
- akshare真实市场数据获取

支持期权类型：欧式、美式、亚式（算术平均/几何平均）

作者: derivatives-pricer team
许可证: MIT
"""

from .black_scholes import (
    bs_call, bs_put, bs_price, implied_vol, vega,
    put_call_parity_check, d1, d2
)
from .greeks import (
    delta, gamma, theta, vega_greek, rho,
    all_greeks, numerical_greeks, greeks_table
)
from .monte_carlo import (
    mc_european, mc_antithetic, mc_control_variate,
    mc_american_ls, mc_asian, convergence_analysis
)
from .finite_difference import (
    fdm_european, fdm_american, fdm_convergence_analysis
)
from .cos_method import (
    cos_european, cos_price, characteristic_function_normal,
    characteristic_function_vg, characteristic_function_nig
)
from .data import (
    get_risk_free_rate, get_option_data_50etf, get_etf_price,
    get_option_chain, build_iv_surface, get_sample_market_params,
    get_realtime_params
)
from .report import generate_html_report, save_html_report

__version__ = "1.0.0"
__all__ = [
    # Black-Scholes
    "bs_call", "bs_put", "bs_price", "implied_vol", "vega",
    "put_call_parity_check", "d1", "d2",
    # Greeks
    "delta", "gamma", "theta", "vega_greek", "rho",
    "all_greeks", "numerical_greeks", "greeks_table",
    # Monte Carlo
    "mc_european", "mc_antithetic", "mc_control_variate",
    "mc_american_ls", "mc_asian", "convergence_analysis",
    # Finite Difference
    "fdm_european", "fdm_american", "fdm_convergence_analysis",
    # COS Method
    "cos_european", "cos_price", "characteristic_function_normal",
    "characteristic_function_vg", "characteristic_function_nig",
    # Market Data
    "get_risk_free_rate", "get_option_data_50etf", "get_etf_price",
    "get_option_chain", "build_iv_surface", "get_sample_market_params",
    "get_realtime_params",
    # Report
    "generate_html_report", "save_html_report",
]
