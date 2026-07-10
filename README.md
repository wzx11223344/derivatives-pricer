# 衍生品定价引擎 (Derivatives Pricer)

专业级衍生品定价工具，实现 Black-Scholes 解析解、蒙特卡洛模拟、有限差分法、COS方法等多种定价方法。

## 特性

- **Black-Scholes 解析解**: 支持连续股息收益率 q 的广义 BS 模型，Newton-Raphson 隐含波动率反推
- **蒙特卡洛模拟**: 标准模拟、对偶变量法、控制变量法、Longstaff-Schwartz 美式期权定价
- **有限差分法**: 显式/隐式/Crank-Nicolson 三种格式，支持美式期权提前行权
- **COS方法**: Fourier 余弦展开 (Fang & Oosterlee 2008)，支持正态/Variance-Gamma/NIG 分布
- **希腊字母**: Delta/Gamma/Theta/Vega/Rho，解析公式 + 数值差分验证
- **真实数据**: 使用 akshare 获取中国市场期权数据和国债收益率
- **HTML报告**: matplotlib 可视化图表，base64 嵌入图片

## 快速开始

### 安装

```bash
cd derivatives-pricer
pip install -r requirements.txt
```

### 命令行使用

```bash
# 欧式看涨期权 (BS方法)
python price.py --type european --option call --spot 3.0 --strike 3.0 --vol 0.2 --r 0.03 --t 0.25

# 美式期权 (蒙特卡洛 Longstaff-Schwartz)
python price.py --type american --method mc --spot 100 --strike 100 --vol 0.3 --r 0.05 --t 1.0

# 亚式期权 (蒙特卡洛)
python price.py --type asian --method mc --spot 50 --strike 50 --vol 0.2 --r 0.03 --t 0.5

# COS方法
python price.py --type european --option put --method cos --spot 50 --strike 50 --vol 0.25 --r 0.04 --t 0.5

# 生成HTML报告
python price.py --type european --option call --spot 3.0 --strike 3.0 --vol 0.2 --r 0.03 --t 0.25 --report

# 使用真实市场数据
python price.py --market-data --method bs

# 隐含波动率反推
python price.py --type european --option call --spot 100 --strike 100 --vol 0.2 --r 0.05 --t 1.0 --iv 10.45

# Put-Call Parity 验证
python price.py --type european --option call --spot 100 --strike 100 --vol 0.2 --r 0.05 --t 1.0 --verify
```

### Python API

```python
from derivatives_pricer import (
    bs_call, bs_put, implied_vol,
    mc_antithetic, mc_american_ls, mc_asian,
    fdm_european, fdm_american,
    cos_european,
    all_greeks, numerical_greeks,
    generate_html_report,
)

# Black-Scholes
price = bs_call(S0=100, K=100, r=0.05, q=0.0, sigma=0.2, T=1.0)

# 蒙特卡洛 (对偶变量法)
price, se = mc_antithetic(S0=100, K=100, r=0.05, q=0.0, sigma=0.2, T=1.0,
                          option_type='call', n_paths=100000)

# 有限差分 (Crank-Nicolson)
price = fdm_european(S0=100, K=100, r=0.05, q=0.0, sigma=0.2, T=1.0,
                     option_type='call', method='cn', M=400, N=400)

# COS方法
price = cos_european(S0=100, K=100, r=0.05, q=0.0, sigma=0.2, T=1.0,
                     option_type='call', N=200)

# 美式期权 (Longstaff-Schwartz)
price = mc_american_ls(S0=100, K=100, r=0.05, q=0.0, sigma=0.3, T=1.0,
                       option_type='put', n_paths=100000, n_steps=50)

# 希腊字母
greeks = all_greeks(S0=100, K=100, r=0.05, q=0.0, sigma=0.2, T=1.0, option_type='call')

# 生成HTML报告
html = generate_html_report(S0=100, K=100, r=0.05, q=0.0, sigma=0.2, T=1.0)
```

## 项目结构

```
derivatives-pricer/
├── price.py                   # CLI 入口
├── derivatives_pricer/
│   ├── __init__.py            # 包初始化
│   ├── black_scholes.py       # Black-Scholes 解析解
│   ├── monte_carlo.py         # 蒙特卡洛模拟
│   ├── finite_difference.py   # 有限差分法
│   ├── cos_method.py          # COS方法 (Fourier余弦展开)
│   ├── greeks.py              # 希腊字母计算
│   ├── data.py                # akshare 真实数据获取
│   └── report.py              # HTML报告生成
├── SKILL.md                   # 技能文档
├── README.md                  # 项目说明
└── requirements.txt           # 依赖
```

## 定价方法对比

| 方法 | 精度 | 速度 | 适用场景 |
|------|------|------|---------|
| Black-Scholes | 精确 | 最快 | 欧式期权，正态分布假设 |
| 蒙特卡洛 (对偶变量) | O(1/sqrt(N)) | 中等 | 路径依赖期权，美式期权 |
| Crank-Nicolson | 二阶精度 | 快 | 欧式/美式期权 |
| COS方法 | 指数收敛 | 快 | 欧式期权，非正态分布 |

## 环境要求

- Python 3.8+
- numpy >= 1.24.0
- scipy >= 1.10.0
- pandas >= 2.0.0
- matplotlib >= 3.7.0
- rich >= 13.0.0
- akshare >= 1.12.0 (可选，用于真实数据)

## 许可证

MIT License
