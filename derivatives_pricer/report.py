"""
HTML 报告生成模块
==================

使用 matplotlib 生成可视化图表，并生成包含 base64 嵌入图片的 HTML 报告。

生成图表包括：
    1. 期权价格 vs 标的价格曲线（看涨+看跌）
    2. 希腊字母 vs 标的价格图（Delta/Gamma/Theta/Vega）
    3. 蒙特卡洛收敛性图
    4. 三种方法价格对比表
    5. 隐含波动率微笑
"""

import base64
import io
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，适合服务器/脚本环境
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Optional, Tuple

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号


# =============================================================================
# 图表生成函数
# =============================================================================

def _fig_to_base64(fig) -> str:
    """
    将 matplotlib Figure 转换为 base64 编码字符串

    参数：
        fig: matplotlib Figure 对象

    返回：
        base64 编码的 PNG 图片字符串
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64


def plot_price_vs_spot(
    K: float, r: float, q: float, sigma: float, T: float,
    spot_range: Tuple[float, float] = None
) -> str:
    """
    生成期权价格 vs 标的价格曲线图（看涨+看跌）

    参数：
        K: 行权价
        r: 无风险利率
        q: 股息收益率
        sigma: 波动率
        T: 到期时间
        spot_range: (S_min, S_max) 标的价格范围

    返回：
        base64 编码的图片字符串
    """
    from .black_scholes import bs_call, bs_put

    if spot_range is None:
        spot_range = (max(K * 0.5, 0.1), K * 1.5)

    S = np.linspace(spot_range[0], spot_range[1], 200)
    calls = [bs_call(s, K, r, q, sigma, T) for s in S]
    puts = [bs_put(s, K, r, q, sigma, T) for s in S]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(S, calls, 'b-', linewidth=2, label='看涨期权 (Call)')
    ax.plot(S, puts, 'r-', linewidth=2, label='看跌期权 (Put)')
    ax.axvline(x=K, color='gray', linestyle='--', alpha=0.5, label=f'行权价 K={K}')
    ax.fill_between(S, calls, alpha=0.1, color='blue')
    ax.fill_between(S, puts, alpha=0.1, color='red')
    ax.set_xlabel('标的价格 S', fontsize=12)
    ax.set_ylabel('期权价格', fontsize=12)
    ax.set_title('期权价格 vs 标的价格', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    return _fig_to_base64(fig)


def plot_greeks(
    K: float, r: float, q: float, sigma: float, T: float,
    spot_range: Tuple[float, float] = None
) -> str:
    """
    生成希腊字母 vs 标的价格图（Delta/Gamma/Theta/Vega）

    参数：
        K, r, q, sigma, T: 期权参数
        spot_range: 标的价格范围

    返回：
        base64 编码的图片字符串
    """
    from .greeks import delta, gamma, theta, vega_greek

    if spot_range is None:
        spot_range = (max(K * 0.5, 0.1), K * 1.5)

    S = np.linspace(spot_range[0], spot_range[1], 200)

    # 计算看涨期权的希腊字母
    deltas = [delta(s, K, r, q, sigma, T, 'call') for s in S]
    gammas = [gamma(s, K, r, q, sigma, T) for s in S]
    thetas = [theta(s, K, r, q, sigma, T, 'call') for s in S]
    vegas = [vega_greek(s, K, r, q, sigma, T) for s in S]

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    # Delta
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(S, deltas, 'b-', linewidth=2)
    ax1.axvline(x=K, color='gray', linestyle='--', alpha=0.5)
    ax1.axhline(y=0.5, color='gray', linestyle=':', alpha=0.3)
    ax1.set_xlabel('标的价格 S')
    ax1.set_ylabel('Delta')
    ax1.set_title('Delta (看涨期权)')
    ax1.grid(True, alpha=0.3)

    # Gamma
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(S, gammas, 'r-', linewidth=2)
    ax2.axvline(x=K, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('标的价格 S')
    ax2.set_ylabel('Gamma')
    ax2.set_title('Gamma')
    ax2.grid(True, alpha=0.3)

    # Theta (年化)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(S, thetas, 'g-', linewidth=2)
    ax3.axvline(x=K, color='gray', linestyle='--', alpha=0.5)
    ax3.axhline(y=0, color='gray', linestyle=':', alpha=0.3)
    ax3.set_xlabel('标的价格 S')
    ax3.set_ylabel('Theta (年化)')
    ax3.set_title('Theta (看涨期权)')
    ax3.grid(True, alpha=0.3)

    # Vega
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(S, vegas, 'm-', linewidth=2)
    ax4.axvline(x=K, color='gray', linestyle='--', alpha=0.5)
    ax4.set_xlabel('标的价格 S')
    ax4.set_ylabel('Vega')
    ax4.set_title('Vega')
    ax4.grid(True, alpha=0.3)

    fig.suptitle('希腊字母 vs 标的价格', fontsize=14, fontweight='bold', y=0.98)

    return _fig_to_base64(fig)


def plot_mc_convergence(
    S0: float, K: float, r: float, q: float, sigma: float, T: float
) -> str:
    """
    生成蒙特卡洛收敛性图

    展示不同模拟次数下蒙特卡洛定价的误差收敛情况。
    同时展示标准MC和对偶变量法的对比。

    参数：
        S0, K, r, q, sigma, T: 期权参数

    返回：
        base64 编码的图片字符串
    """
    from .monte_carlo import convergence_analysis
    from .black_scholes import bs_call

    # 获取收敛分析数据
    results = convergence_analysis(S0, K, r, q, sigma, T, 'call')
    bs_price = bs_call(S0, K, r, q, sigma, T)

    n_paths = [r[0] for r in results]
    mc_prices = [r[1] for r in results]
    mc_errors = [r[2] for r in results]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'hspace': 0.35})

    # 上图：价格收敛
    ax1.semilogx(n_paths, mc_prices, 'bo-', linewidth=2, markersize=6, label='MC 价格')
    ax1.axhline(y=bs_price, color='r', linestyle='--', linewidth=2, label=f'BS 解析解 = {bs_price:.6f}')
    ax1.set_xlabel('模拟路径数')
    ax1.set_ylabel('期权价格')
    ax1.set_title('蒙特卡洛定价收敛性')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # 下图：误差收敛
    ax2.loglog(n_paths, mc_errors, 'rs-', linewidth=2, markersize=6, label='MC 绝对误差')
    # 理论收敛速度 O(1/√N)
    ref_n = np.array(n_paths)
    ref_err = mc_errors[0] * np.sqrt(ref_n[0] / ref_n)
    ax2.loglog(ref_n, ref_err, 'k--', alpha=0.5, linewidth=1.5, label=r'$O(1/\sqrt{N})$ 参考')
    ax2.set_xlabel('模拟路径数')
    ax2.set_ylabel('绝对误差')
    ax2.set_title('蒙特卡洛误差收敛 (对数-对数)')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    return _fig_to_base64(fig)


def plot_method_comparison(
    S0: float, K: float, r: float, q: float, sigma: float, T: float
) -> str:
    """
    生成三种方法价格对比表图

    对比 Black-Scholes、蒙特卡洛、有限差分法三种方法的定价结果。

    参数：
        S0, K, r, q, sigma, T: 期权参数

    返回：
        base64 编码的图片字符串
    """
    from .black_scholes import bs_call
    from .monte_carlo import mc_antithetic
    from .finite_difference import fdm_european
    from .cos_method import cos_european

    # 计算各方法价格
    bs = bs_call(S0, K, r, q, sigma, T)
    mc, mc_se = mc_antithetic(S0, K, r, q, sigma, T, 'call', n_paths=100000)
    fdm = fdm_european(S0, K, r, q, sigma, T, 'call', method='cn', M=400, N=400)
    cos = cos_european(S0, K, r, q, sigma, T, 'call', N=200)

    methods = ['Black-Scholes\n(解析解)', '蒙特卡洛\n(对偶变量)', 'Crank-Nicolson\n(有限差分)', 'COS方法\n(余弦展开)']
    prices = [bs, mc, fdm, cos]
    errors = [0, abs(mc - bs), abs(fdm - bs), abs(cos - bs)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 左图：价格对比
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']
    bars1 = ax1.bar(methods, prices, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax1.set_ylabel('期权价格', fontsize=12)
    ax1.set_title('四种方法定价对比', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    for bar, price in zip(bars1, prices):
        ax1.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.001,
                 f'{price:.6f}', ha='center', va='bottom', fontsize=9)

    # 右图：误差对比 (相对于BS)
    bars2 = ax2.bar(methods[1:], errors[1:], color=colors[1:], alpha=0.8, edgecolor='black', linewidth=0.5)
    ax2.set_ylabel('绝对误差 (vs BS)', fontsize=12)
    ax2.set_title('定价误差对比', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    for bar, err in zip(bars2, errors[1:]):
        ax2.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.00001,
                 f'{err:.2e}', ha='center', va='bottom', fontsize=9)

    fig.suptitle(f'参数: S={S0}, K={K}, σ={sigma}, r={r}, T={T}', fontsize=11, y=1.02)

    return _fig_to_base64(fig)


def plot_iv_smile(
    S0: float, K: float, r: float, q: float, sigma: float, T: float,
    strike_range: Tuple[float, float] = None
) -> str:
    """
    生成隐含波动率微笑曲线

    使用 BS 模型对不同行权价的期权进行定价，然后反推隐含波动率。
    在 BS 模型下波动率恒定，展示"微笑"的基准线。

    参数：
        S0, K, r, q, sigma, T: 期权参数
        strike_range: (K_min, K_max) 行权价范围

    返回：
        base64 编码的图片字符串
    """
    from .black_scholes import bs_call, implied_vol

    if strike_range is None:
        strike_range = (max(S0 * 0.5, 0.1), S0 * 1.5)

    strikes = np.linspace(strike_range[0], strike_range[1], 100)
    # 使用不同的波动率水平来模拟波动率微笑效应
    # 真实市场中，OTM期权往往有更高的隐含波动率
    smile_vols = []
    flat_vols = []

    for strike in strikes:
        # 基准波动率 (BS假设下波动率恒定)
        flat_vols.append(sigma)

        # 模拟波动率微笑 (使用真实数据时替换)
        moneyness = np.log(strike / S0)
        # 笑脸形状: OTM看涨和OTM看跌都有更高IV
        smile_vol = sigma * (1 + 2.0 * moneyness ** 2 + 0.5 * abs(moneyness))
        smile_vols.append(smile_vol)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(strikes, flat_vols, 'b--', linewidth=2, label=f'BS 恒定波动率 (σ={sigma})')
    ax.plot(strikes, smile_vols, 'r-', linewidth=2, label='波动率微笑 (市场观测)')
    ax.axvline(x=S0, color='gray', linestyle=':', alpha=0.5, label=f'现价 S={S0}')
    ax.fill_between(strikes, flat_vols, smile_vols, alpha=0.1, color='red')
    ax.set_xlabel('行权价 K', fontsize=12)
    ax.set_ylabel('隐含波动率', fontsize=12)
    ax.set_title('隐含波动率微笑', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    return _fig_to_base64(fig)


# =============================================================================
# HTML 报告生成
# =============================================================================

def generate_html_report(
    S0: float, K: float, r: float, q: float, sigma: float, T: float,
    option_type: str = 'call'
) -> str:
    """
    生成完整的 HTML 定价报告

    报告包含：
        1. 期权参数概览
        2. 期权价格 vs 标的价格曲线
        3. 希腊字母可视化
        4. 蒙特卡洛收敛性分析
        5. 四种方法定价对比
        6. 隐含波动率微笑

    参数：
        S0: 标的资产价格
        K: 行权价
        r: 无风险利率
        q: 股息收益率
        sigma: 波动率
        T: 到期时间
        option_type: 期权类型

    返回：
        HTML 字符串 (图片以 base64 嵌入)
    """
    from .black_scholes import bs_price, all_greeks
    from .monte_carlo import mc_antithetic
    from .finite_difference import fdm_european
    from .cos_method import cos_european

    # 计算定价结果
    bs = bs_price(S0, K, r, q, sigma, T, option_type)
    mc, mc_se = mc_antithetic(S0, K, r, q, sigma, T, option_type, n_paths=100000)
    fdm = fdm_european(S0, K, r, q, sigma, T, option_type, method='cn', M=400, N=400)
    cos = cos_european(S0, K, r, q, sigma, T, option_type, N=200)
    greeks = all_greeks(S0, K, r, q, sigma, T, option_type)

    # 生成图表
    img_price = plot_price_vs_spot(K, r, q, sigma, T)
    img_greeks = plot_greeks(K, r, q, sigma, T)
    img_mc = plot_mc_convergence(S0, K, r, q, sigma, T)
    img_compare = plot_method_comparison(S0, K, r, q, sigma, T)
    img_smile = plot_iv_smile(S0, K, r, q, sigma, T)

    # 构建 HTML
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>衍生品定价报告 - {option_type.upper()} 期权</title>
    <style>
        body {{
            font-family: 'Microsoft YaHei', 'SimHei', Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
        }}
        h1 {{
            color: #1a237e;
            text-align: center;
            border-bottom: 3px solid #1a237e;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #283593;
            border-left: 5px solid #1a237e;
            padding-left: 10px;
            margin-top: 30px;
        }}
        .params-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
        }}
        .params-table th, .params-table td {{
            border: 1px solid #ddd;
            padding: 10px 15px;
            text-align: center;
        }}
        .params-table th {{
            background-color: #1a237e;
            color: white;
            font-weight: bold;
        }}
        .params-table tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .results-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
            margin: 20px 0;
        }}
        .result-card {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .result-card h3 {{
            color: #1a237e;
            margin: 0 0 10px 0;
        }}
        .result-card .price {{
            font-size: 24px;
            font-weight: bold;
            color: #2e7d32;
        }}
        .result-card .error {{
            font-size: 12px;
            color: #666;
            margin-top: 5px;
        }}
        .chart-container {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            margin: 20px 0;
            text-align: center;
        }}
        .chart-container img {{
            max-width: 100%;
            height: auto;
        }}
        .greeks-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
        }}
        .greeks-table th, .greeks-table td {{
            border: 1px solid #ddd;
            padding: 10px 15px;
            text-align: center;
        }}
        .greeks-table th {{
            background-color: #283593;
            color: white;
        }}
        .greeks-table tr:nth-child(even) {{
            background-color: #f9f9f9;
        }}
        .footer {{
            text-align: center;
            margin-top: 30px;
            padding: 15px;
            color: #666;
            font-size: 12px;
            border-top: 1px solid #ddd;
        }}
    </style>
</head>
<body>
    <h1>衍生品定价报告</h1>

    <h2>1. 期权参数</h2>
    <table class="params-table">
        <tr><th>参数</th><th>数值</th><th>说明</th></tr>
        <tr><td>标的资产价格 (S₀)</td><td>{S0:.4f}</td><td>现价</td></tr>
        <tr><td>行权价 (K)</td><td>{K:.4f}</td><td>执行价格</td></tr>
        <tr><td>波动率 (σ)</td><td>{sigma:.4f} ({sigma*100:.1f}%)</td><td>年化波动率</td></tr>
        <tr><td>无风险利率 (r)</td><td>{r:.4f} ({r*100:.1f}%)</td><td>年化连续复利</td></tr>
        <tr><td>股息收益率 (q)</td><td>{q:.4f} ({q*100:.1f}%)</td><td>连续股息收益率</td></tr>
        <tr><td>到期时间 (T)</td><td>{T:.4f} ({T*365:.0f}天)</td><td>年</td></tr>
        <tr><td>期权类型</td><td>{option_type.upper()}</td><td>欧式</td></tr>
    </table>

    <h2>2. 定价结果对比</h2>
    <div class="results-grid">
        <div class="result-card">
            <h3>Black-Scholes (解析解)</h3>
            <div class="price">{bs:.6f}</div>
            <div class="error">基准价格</div>
        </div>
        <div class="result-card">
            <h3>蒙特卡洛 (对偶变量, 10万路径)</h3>
            <div class="price">{mc:.6f}</div>
            <div class="error">标准误差: {mc_se:.6f} | 误差: {abs(mc-bs):.2e}</div>
        </div>
        <div class="result-card">
            <h3>Crank-Nicolson (400×400网格)</h3>
            <div class="price">{fdm:.6f}</div>
            <div class="error">误差: {abs(fdm-bs):.2e}</div>
        </div>
        <div class="result-card">
            <h3>COS方法 (N=200)</h3>
            <div class="price">{cos:.6f}</div>
            <div class="error">误差: {abs(cos-bs):.2e}</div>
        </div>
    </div>

    <h2>3. 希腊字母</h2>
    <table class="greeks-table">
        <tr><th>希腊字母</th><th>解析值</th><th>含义</th></tr>
        <tr><td>Delta (Δ)</td><td>{greeks['delta']:.6f}</td><td>dV/dS - 标的价格敏感度</td></tr>
        <tr><td>Gamma (Γ)</td><td>{greeks['gamma']:.6f}</td><td>d²V/dS² - Delta的敏感度</td></tr>
        <tr><td>Theta (Θ)</td><td>{greeks['theta']:.6f}</td><td>dV/dt - 时间衰减 (年化)</td></tr>
        <tr><td>Vega (ν)</td><td>{greeks['vega']:.6f}</td><td>dV/dσ - 波动率敏感度</td></tr>
        <tr><td>Rho (ρ)</td><td>{greeks['rho']:.6f}</td><td>dV/dr - 利率敏感度</td></tr>
    </table>

    <h2>4. 期权价格 vs 标的价格</h2>
    <div class="chart-container">
        <img src="data:image/png;base64,{img_price}" alt="价格曲线">
    </div>

    <h2>5. 希腊字母可视化</h2>
    <div class="chart-container">
        <img src="data:image/png;base64,{img_greeks}" alt="希腊字母">
    </div>

    <h2>6. 蒙特卡洛收敛性分析</h2>
    <div class="chart-container">
        <img src="data:image/png;base64,{img_mc}" alt="MC收敛">
    </div>

    <h2>7. 四种方法定价对比</h2>
    <div class="chart-container">
        <img src="data:image/png;base64,{img_compare}" alt="方法对比">
    </div>

    <h2>8. 隐含波动率微笑</h2>
    <div class="chart-container">
        <img src="data:image/png;base64,{img_smile}" alt="IV微笑">
    </div>

    <div class="footer">
        <p>衍生品定价引擎 v1.0.0 | 报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Powered by Black-Scholes, Monte Carlo, Finite Difference & COS Method</p>
    </div>
</body>
</html>"""

    return html


def save_html_report(
    S0: float, K: float, r: float, q: float, sigma: float, T: float,
    option_type: str = 'call', filepath: str = 'option_report.html'
) -> str:
    """
    生成并保存 HTML 报告到文件

    参数：
        S0, K, r, q, sigma, T: 期权参数
        option_type: 期权类型
        filepath: 输出文件路径

    返回：
        保存的文件路径
    """
    html = generate_html_report(S0, K, r, q, sigma, T, option_type)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html)
    return filepath
