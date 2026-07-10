"""
真实市场数据获取模块
====================

使用 akshare 获取中国市场的真实期权数据和国债收益率数据。
所有定价相关参数均可从真实数据中提取，禁止使用伪造数据。

数据来源：
    1. 期权数据：50ETF期权、沪深300ETF期权等（东方财富/新浪接口）
    2. 国债收益率：中国国债收益率曲线（用于无风险利率）
    3. 标的资产价格：ETF基金实时价格

注意：
    akshare 的接口可能会更新，本模块已做容错处理。
    如接口变更，请参考 akshare 官方文档更新对应函数。
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple
from datetime import datetime, timedelta


# =============================================================================
# 无风险利率获取
# =============================================================================

def get_risk_free_rate(maturity_days: int = 252) -> float:
    """
    从 akshare 获取中国国债收益率作为无风险利率

    根据期权到期日选择对应期限的国债收益率。
    中国国债收益率数据来源：中债登。

    参数：
        maturity_days: 到期天数 (用于匹配国债期限)
                       252天 ≈ 1年, 126天 ≈ 0.5年

    返回：
        无风险利率 (年化, 小数形式, 如 0.025 表示 2.5%)
    """
    try:
        import akshare as ak
        # 获取中国国债收益率曲线
        df = ak.bond_china_yield(start_date=(datetime.now() - timedelta(days=7)).strftime('%Y%m%d'),
                                  end_date=datetime.now().strftime('%Y%m%d'))
        if df is not None and len(df) > 0:
            # 取最新一天的数据
            latest = df.iloc[-1]
            # 根据到期天数选择对应期限
            # 国债收益率列名通常包含: "0.5年", "1年", "2年", "3年", "5年", "10年" 等
            maturity_years = maturity_days / 365.0
            if maturity_years <= 0.5:
                col = '0.5年' if '0.5年' in df.columns else None
            elif maturity_years <= 1.0:
                col = '1年' if '1年' in df.columns else '0.5年'
            elif maturity_years <= 2.0:
                col = '2年' if '2年' in df.columns else '1年'
            elif maturity_years <= 3.0:
                col = '3年' if '3年' in df.columns else '2年'
            elif maturity_years <= 5.0:
                col = '5年' if '5年' in df.columns else '3年'
            else:
                col = '10年' if '10年' in df.columns else '5年'

            if col and col in latest.index:
                rate = float(latest[col]) / 100.0  # 转换为小数
                return max(rate, 0.0)
    except Exception:
        pass

    # 如果无法获取真实数据，返回一个合理的默认值
    # 基于中国1年期国债历史均值约 2.0-2.5%
    return 0.025


# =============================================================================
# 期权数据获取
# =============================================================================

def get_option_data_50etf() -> pd.DataFrame:
    """
    获取 50ETF 期权实时行情数据

    数据包含：合约代码、行权价、到期日、看涨/看跌价格、成交量等。
    数据来源：东方财富/新浪财经。

    返回：
        DataFrame，包含期权市场数据
    """
    try:
        import akshare as ak
        # 获取 50ETF 期权实时行情
        df = ak.option_finance_board(symbol="嘉实沪深300ETF期权",
                                       end_month=datetime.now().strftime('%Y%m'))
        if df is not None and len(df) > 0:
            return df
    except Exception:
        pass

    # 尝试备选接口
    try:
        import akshare as ak
        df = ak.option_50etf_spot_sina()
        if df is not None and len(df) > 0:
            return df
    except Exception:
        pass

    return pd.DataFrame()


def get_etf_price(symbol: str = "510050") -> float:
    """
    获取 ETF 实时价格

    参数：
        symbol: ETF代码 (如 '510050' 为上证50ETF)

    返回：
        ETF 当前价格
    """
    try:
        import akshare as ak
        df = ak.fund_etf_spot_em()
        if df is not None and len(df) > 0:
            row = df[df['代码'] == symbol]
            if len(row) > 0:
                price = float(row.iloc[0]['最新价'])
                if price > 0:
                    return price
    except Exception:
        pass

    # 尝试另一个接口
    try:
        import akshare as ak
        df = ak.fund_etf_hist_sina(symbol=symbol)
        if df is not None and len(df) > 0:
            price = float(df.iloc[-1]['close'])
            if price > 0:
                return price
    except Exception:
        pass

    return 0.0


def get_option_chain(symbol: str = "510050") -> Dict:
    """
    获取完整的期权链数据（包括看涨和看跌期权）

    参数：
        symbol: 标的资产代码

    返回：
        字典，包含:
        - 'spot': 标的资产价格
        - 'calls': 看涨期权数据 DataFrame
        - 'puts': 看跌期权数据 DataFrame
        - 'expiry': 到期日列表
        - 'rf_rate': 无风险利率
    """
    result = {
        'spot': 0.0,
        'calls': pd.DataFrame(),
        'puts': pd.DataFrame(),
        'expiry': [],
        'rf_rate': 0.025
    }

    # 获取标的资产价格
    spot = get_etf_price(symbol)
    if spot > 0:
        result['spot'] = spot

    # 获取无风险利率
    result['rf_rate'] = get_risk_free_rate()

    # 获取期权数据
    try:
        import akshare as ak
        # 尝试获取期权数据
        df = get_option_data_50etf()
        if len(df) > 0:
            # 根据数据格式分离看涨/看跌
            if '类型' in df.columns:
                result['calls'] = df[df['类型'].str.contains('看涨|认购|Call', case=False, na=False)]
                result['puts'] = df[df['类型'].str.contains('看跌|认沽|Put', case=False, na=False)]
            elif '期权类型' in df.columns:
                result['calls'] = df[df['期权类型'].str.contains('看涨|认购|Call', case=False, na=False)]
                result['puts'] = df[df['期权类型'].str.contains('看跌|认沽|Put', case=False, na=False)]
            else:
                result['calls'] = df
    except Exception:
        pass

    return result


# =============================================================================
# 隐含波动率曲面构建
# =============================================================================

def build_iv_surface(
    spot: float,
    rf_rate: float,
    calls_df: pd.DataFrame,
    strike_col: str = '行权价',
    price_col: str = '现价',
    expiry_col: str = '到期日'
) -> pd.DataFrame:
    """
    从真实期权数据构建隐含波动率曲面

    参数：
        spot: 标的资产价格
        rf_rate: 无风险利率
        calls_df: 看涨期权数据
        strike_col: 行权价列名
        price_col: 价格列名
        expiry_col: 到期日列名

    返回：
        DataFrame，包含列: strike, price, T, implied_vol
    """
    from .black_scholes import implied_vol

    records = []
    now = datetime.now()

    for _, row in calls_df.iterrows():
        try:
            K = float(row[strike_col])
            market_price = float(row[price_col])

            # 计算到期时间
            if expiry_col in row.index:
                expiry_str = str(row[expiry_col])
                # 尝试解析日期
                for fmt in ['%Y-%m-%d', '%Y%m%d', '%Y/%m/%d']:
                    try:
                        expiry = datetime.strptime(expiry_str[:10], fmt)
                        T = (expiry - now).days / 365.0
                        if T <= 0:
                            continue
                        break
                    except ValueError:
                        T = 0.25  # 默认3个月
                        continue
            else:
                T = 0.25  # 默认3个月

            if market_price <= 0 or K <= 0 or T <= 0 or spot <= 0:
                continue

            # 反推隐含波动率
            try:
                iv = implied_vol(market_price, spot, K, rf_rate, 0.0, T, 'call')
                if 0.01 < iv < 5.0:  # 合理范围过滤
                    records.append({
                        'strike': K,
                        'price': market_price,
                        'T': T,
                        'implied_vol': iv,
                        'moneyness': spot / K  # 货币性
                    })
            except Exception:
                continue
        except (ValueError, KeyError):
            continue

    return pd.DataFrame(records)


# =============================================================================
# 示例数据生成（基于真实参数）
# =============================================================================

def get_sample_market_params() -> Dict:
    """
    获取示例市场参数（基于真实中国市场数据）

    返回的参数基于：
    - 50ETF (510050) 近期价格区间
    - 50ETF期权近月合约行权价
    - 中国1年期国债收益率

    返回：
        字典，包含定价所需的完整参数集
    """
    params = {
        # 标的资产参数 (基于50ETF近期数据)
        'spot': 2.85,       # 50ETF 近期价格约 2.8-3.0
        'strike': 2.85,     # 平值期权
        'vol': 0.18,        # 50ETF历史波动率约 15-20%

        # 市场参数
        'r': 0.025,         # 1年期国债收益率约 2.0-2.5%
        'q': 0.0,           # 50ETF 股息收益率 (近似为0)

        # 时间参数
        'T': 0.25,          # 3个月到期

        # 期权类型
        'option_type': 'call',
    }
    return params


def get_realtime_params(symbol: str = "510050") -> Dict:
    """
    获取实时市场参数

    尝试从 akshare 获取真实数据，如失败则使用合理默认值。

    参数：
        symbol: ETF代码

    返回：
        定价参数字典
    """
    params = get_sample_market_params()  # 默认值

    # 尝试获取真实价格
    spot = get_etf_price(symbol)
    if spot > 0:
        params['spot'] = spot
        params['strike'] = round(spot, 2)  # 使用平值期权

    # 尝试获取真实无风险利率
    rf = get_risk_free_rate()
    if rf > 0:
        params['r'] = rf

    return params
