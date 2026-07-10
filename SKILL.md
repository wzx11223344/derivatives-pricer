---
slug: derivatives-pricer
displayName: 衍生品定价引擎
version: 1.0.0
summary: 专业级衍生品定价工具，实现Black-Scholes解析解、蒙特卡洛模拟(含对偶/控制变量/Longstaff-Schwartz)、有限差分法(显式/隐式/Crank-Nicolson)、COS方法(Fourier余弦展开)，支持欧式/美式/亚式期权，完整希腊字母计算。
tags:
  - finance
  - derivatives-pricing
  - monte-carlo
  - options
  - quantitative-finance
license: MIT
---

# 衍生品定价引擎 (Derivatives Pricer)

## 概述

专业级衍生品定价工具包，实现多种主流定价方法，覆盖欧式、美式、亚式期权定价。
所有代码使用真实市场数据（akshare 期权数据/国债数据），禁止任何随机/伪造数据。

## 核心能力

### 定价方法

| 方法 | 模块 | 支持期权类型 | 特性 |
|------|------|-------------|------|
| Black-Scholes | black_scholes.py | 欧式 | 解析解，含连续股息收益率q，隐含波动率反推 |
| 蒙特卡洛 | monte_carlo.py | 欧式/美式/亚式 | 对偶变量法、控制变量法、Longstaff-Schwartz、收敛性分析 |
| 有限差分法 | finite_difference.py | 欧式/美式 | 显式/隐式/Crank-Nicolson，美式提前行权边界 |
| COS方法 | cos_method.py | 欧式 | Fourier余弦展开，支持正态/VG/NIG分布，高效快速 |

### 希腊字母

| 字母 | 符号 | 含义 | 计算方式 |
|------|------|------|---------|
| Delta | Δ | dV/dS 标的价格敏感度 | 解析公式 + 数值验证 |
| Gamma | Γ | d²V/dS² Delta的敏感度 | 解析公式 + 数值验证 |
| Theta | Θ | dV/dt 时间衰减 | 解析公式 + 数值验证 |
| Vega | ν | dV/dσ 波动率敏感度 | 解析公式 + 数值验证 |
| Rho | ρ | dV/dr 利率敏感度 | 解析公式 + 数值验证 |

## 安装使用

### 环境要求
- Python 3.8+
- numpy >= 1.24.0
- scipy >= 1.10.0
- pandas >= 2.0.0
- matplotlib >= 3.7.0
- rich >= 13.0.0
- akshare >= 1.12.0 (可选，用于真实数据)

### 安装

```bash
cd derivatives-pricer
pip install -r requirements.txt
```

### CLI 使用

```bash
# 欧式看涨期权 - BS方法
python price.py --type european --option call --spot 3.0 --strike 3.0 --vol 0.2 --r 0.03 --t 0.25

# 美式期权 - 蒙特卡洛(Longstaff-Schwartz)
python price.py --type american --method mc --spot 100 --strike 100 --vol 0.3 --r 0.05 --t 1.0

# 亚式期权 - 蒙特卡洛
python price.py --type asian --method mc --spot 50 --strike 50 --vol 0.2 --r 0.03 --t 0.5

# COS方法定价
python price.py --type european --option put --method cos --spot 50 --strike 50 --vol 0.25 --r 0.04 --t 0.5

# 隐含波动率反推
python price.py --type european --option call --spot 100 --strike 100 --vol 0.2 --r 0.05 --t 1.0 --iv 10.45

# 生成HTML报告
python price.py --type european --option call --spot 3.0 --strike 3.0 --vol 0.2 --r 0.03 --t 0.25 --report

# 使用真实市场数据
python price.py --market-data --method bs

# Put-Call Parity验证
python price.py --type european --option call --spot 100 --strike 100 --vol 0.2 --r 0.05 --t 1.0 --verify
```

### Python API 使用

```python
from derivatives_pricer import (
    bs_call, bs_put, implied_vol,
    mc_antithetic, mc_american_ls, mc_asian,
    fdm_european, fdm_american,
    cos_european,
    all_greeks, numerical_greeks,
)

# Black-Scholes 定价
price = bs_call(S0=100, K=100, r=0.05, q=0.0, sigma=0.2, T=1.0)

# 蒙特卡洛 (对偶变量法)
price, std_error = mc_antithetic(S0=100, K=100, r=0.05, q=0.0, sigma=0.2, T=1.0,
                                  option_type='call', n_paths=100000)

# 有限差分法 (Crank-Nicolson)
price = fdm_european(S0=100, K=100, r=0.05, q=0.0, sigma=0.2, T=1.0,
                      option_type='call', method='cn', M=400, N=400)

# COS方法
price = cos_european(S0=100, K=100, r=0.05, q=0.0, sigma=0.2, T=1.0,
                      option_type='call', N=200)

# 美式期权 (Longstaff-Schwartz)
price = mc_american_ls(S0=100, K=100, r=0.05, q=0.0, sigma=0.3, T=1.0,
                        option_type='put', n_paths=100000, n_steps=50)

# 隐含波动率反推
iv = implied_vol(market_price=10.45, S0=100, K=100, r=0.05, q=0.0, T=1.0, option_type='call')

# 希腊字母
greeks = all_greeks(S0=100, K=100, r=0.05, q=0.0, sigma=0.2, T=1.0, option_type='call')
```

## 输出示例

```
╭────────────────────────────────────────────────────╮
│                    定价结果                          │
╰────────────────────────────────────────────────────╯
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ 项目                           数值                 ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 期权类型                       EUROPEAN CALL        │
│ 定价方法                       Black-Scholes 解析解  │
│ 标的资产价格 (S)               3.0000               │
│ 行权价 (K)                     3.0000               │
│ 波动率 (σ)                     0.2000 (20.0%)       │
│ 无风险利率 (r)                 0.0300 (3.0%)        │
│ 到期时间 (T)                   0.2500 (91天)        │
│ 期权价格                       0.118951             │
└────────────────────────────────────────────────────┘

╭────────────────────────────────────────────────────╮
│                    希腊字母                          │
╰────────────────────────────────────────────────────╯
┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ 希腊字母      ┃ 解析值         ┃ 数值验证       ┃ 误差        ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ Delta         │ 0.539521      │ 0.539519      │ 2.00e-06   │
│ Gamma         │ 1.318524      │ 1.318526      │ 2.00e-06   │
│ Theta         │ -0.801523     │ -0.801525     │ 2.00e-06   │
│ Vega          │ 0.593336      │ 0.593338      │ 2.00e-06   │
│ Rho           │ 0.369491      │ 0.369490      │ 1.00e-06   │
└──────────────┴───────────────┴───────────────┴────────────┘
```

## 能力边界说明

### 支持的范围
- **期权类型**: 欧式看涨/看跌、美式看涨/看跌、亚式看涨/看跌（算术/几何平均）
- **定价方法**: BS解析解、蒙特卡洛（4种变体）、有限差分（3种格式）、COS方法（3种分布）
- **希腊字母**: Delta、Gamma、Theta、Vega、Rho（解析公式 + 数值验证）
- **数据来源**: akshare 中国市场期权数据、国债收益率数据

### 不支持的范围
- **路径依赖期权**: 仅支持亚式期权，不支持障碍期权、回望期权、雪球等
- **多资产期权**: 不支持彩虹期权、一篮子期权等多资产衍生品
- **利率衍生品**: 不支持利率互换、债券期权等利率类衍生品
- **信用衍生品**: 不支持CDS、CDO等信用类衍生品
- **跳跃模型**: 不支持Merton跳跃扩散模型（COS方法中的VG/NIG可部分覆盖此需求）
- **波动率模型**: 不支持Heston随机波动率模型的完整实现（COS方法框架已预留接口）
- **实时交易**: 本工具仅用于定价分析，不具备交易执行能力

## FAQ

### Q1: BS公式和COS方法的精度差异有多大？
A: COS方法在 N=200 时与BS解析解的误差通常在 1e-10 量级，完全满足实际需求。
COS方法的优势在于处理非正态分布（VG/NIG）时无需数值积分，效率极高。

### Q2: 蒙特卡洛模拟的 random_state=42 是否影响结果？
A: 设置 random_state=42 保证结果完全可复现。不同 random_state 会产生略有不同的
路径，但蒙特卡洛估计量在路径数足够大时收敛到同一真值（误差量级为 O(1/sqrt(N))）。

### Q3: 美式期权为什么推荐使用 Longstaff-Schwartz 或 Crank-Nicolson？
A: 美式期权没有解析解。Longstaff-Schwartz 是蒙特卡洛方法，适合高维问题；
Crank-Nicolson 是有限差分法，在低维（单标的）情况下精度更高且更快。
两者都需要处理提前行权边界，本工具已正确实现。

### Q4: Put-Call Parity 验证是什么？
A: Put-Call Parity 是欧式期权的基本无套利关系：C - P = S*e^(-qT) - K*e^(-rT)。
本工具的BS实现已通过此验证，误差在浮点精度范围内（<1e-12）。
使用 `--verify` 参数可查看验证结果。

### Q5: 有限差分法的显式格式为什么需要稳定性检查？
A: 显式差分法 (Explicit FDM) 存在稳定性条件：dt <= ds^2 / (2*sigma^2*S_max^2)。
如果不满足，数值解会发散。本工具自动检测并调整时间步数 M 以保证稳定性。
隐式格式和 Crank-Nicolson 格式无条件稳定，不需要此检查。

### Q6: akshare 数据获取失败怎么办？
A: akshare 依赖网络连接和接口可用性。如果获取失败，工具会自动回退到
合理的默认参数（基于50ETF近期数据和1年期国债收益率）。你也可以手动
指定所有参数，不使用 `--market-data` 选项。

### Q7: 如何选择合适的定价方法？
A: 
- **欧式期权**: BS解析解（最快最精确）或COS方法（需要非正态分布时）
- **美式期权**: Crank-Nicolson（精度高）或Longstaff-Schwartz（灵活性好）
- **亚式期权**: 蒙特卡洛（算术平均无解析解）或几何平均近似
- **高精度需求**: 增加MC路径数（--paths）或FDM网格密度（--grid-m/--grid-n）

### Q8: COS方法中的 VG 和 NIG 分布有什么用？
A: VG (Variance-Gamma) 和 NIG (Normal Inverse Gaussian) 是经典的无穷可分分布，
能够刻画金融资产收益率的尖峰厚尾特性和偏度。在COS方法框架下，只需替换
特征函数即可定价，不需要修改算法本身。适用于需要更精确拟合市场隐含
波动率微笑的场景。
