#!/usr/bin/env python
"""
衍生品定价引擎 - CLI 入口
==========================

命令行接口，支持多种定价方法和期权类型。

使用示例：
    # 欧式看涨期权 - BS方法
    python price.py --type european --option call --spot 3.0 --strike 3.0 --vol 0.2 --r 0.03 --t 0.25

    # 美式期权 - 蒙特卡洛(Longstaff-Schwartz)
    python price.py --type american --method mc --spot 100 --strike 100 --vol 0.3 --r 0.05 --t 1.0

    # 亚式期权 - 蒙特卡洛
    python price.py --type asian --method mc --spot 50 --strike 50

    # 生成HTML报告
    python price.py --type european --option call --spot 3.0 --strike 3.0 --vol 0.2 --r 0.03 --t 0.25 --report

    # 使用真实市场数据
    python price.py --market-data --method bs
"""

import argparse
import sys
import os

# 将当前目录加入路径，确保可以导入 derivatives_pricer 包
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from derivatives_pricer import (
    # Black-Scholes
    bs_call, bs_put, bs_price, implied_vol, put_call_parity_check,
    # Greeks
    all_greeks, numerical_greeks,
    # Monte Carlo
    mc_european, mc_antithetic, mc_control_variate, mc_american_ls, mc_asian,
    # Finite Difference
    fdm_european, fdm_american,
    # COS Method
    cos_european,
)


def main():
    """主函数：解析命令行参数并执行定价"""
    parser = argparse.ArgumentParser(
        description='衍生品定价引擎 - 支持BS解析解、蒙特卡洛、有限差分、COS方法',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 欧式看涨期权 (BS方法)
  python price.py --type european --option call --spot 3.0 --strike 3.0 --vol 0.2 --r 0.03 --t 0.25

  # 美式期权 (蒙特卡洛 Longstaff-Schwartz)
  python price.py --type american --method mc --spot 100 --strike 100 --vol 0.3 --r 0.05 --t 1.0

  # 亚式期权 (蒙特卡洛)
  python price.py --type asian --method mc --spot 50 --strike 50 --vol 0.2 --r 0.03 --t 0.5

  # 欧式期权 (COS方法)
  python price.py --type european --option put --method cos --spot 50 --strike 50 --vol 0.25 --r 0.04 --t 0.5

  # 生成HTML报告
  python price.py --type european --option call --spot 3.0 --strike 3.0 --vol 0.2 --r 0.03 --t 0.25 --report
        """
    )

    # 期权类型
    parser.add_argument('--type', dest='opt_type', default='european',
                        choices=['european', 'american', 'asian'],
                        help='期权类型: european(欧式), american(美式), asian(亚式) [默认: european]')

    # 期权方向
    parser.add_argument('--option', dest='option_type', default='call',
                        choices=['call', 'put'],
                        help='期权方向: call(看涨), put(看跌) [默认: call]')

    # 定价方法
    parser.add_argument('--method', dest='method', default='bs',
                        choices=['bs', 'mc', 'mc-cv', 'fdm', 'cn', 'cos'],
                        help='定价方法: bs(Black-Scholes), mc(蒙特卡洛对偶), '
                             'mc-cv(蒙特卡洛控制变量), fdm(显式有限差分), '
                             'cn(Crank-Nicolson), cos(COS方法) [默认: bs]')

    # 定价参数
    parser.add_argument('--spot', type=float, default=100.0, help='标的资产价格 [默认: 100]')
    parser.add_argument('--strike', type=float, default=100.0, help='行权价 [默认: 100]')
    parser.add_argument('--vol', type=float, default=0.2, help='波动率 (年化) [默认: 0.2]')
    parser.add_argument('--r', type=float, default=0.05, help='无风险利率 [默认: 0.05]')
    parser.add_argument('--q', type=float, default=0.0, help='连续股息收益率 [默认: 0.0]')
    parser.add_argument('--t', type=float, default=1.0, help='到期时间 (年) [默认: 1.0]')

    # 蒙特卡洛参数
    parser.add_argument('--paths', type=int, default=100000,
                        help='蒙特卡洛模拟路径数 [默认: 100000]')

    # 有限差分参数
    parser.add_argument('--grid-m', type=int, default=200,
                        help='有限差分价格网格点数 [默认: 200]')
    parser.add_argument('--grid-n', type=int, default=200,
                        help='有限差分时间网格点数 [默认: 200]')

    # COS方法参数
    parser.add_argument('--cos-n', type=int, default=200,
                        help='COS方法余弦展开项数 [默认: 200]')

    # 隐含波动率
    parser.add_argument('--iv', type=float, default=None,
                        help='隐含波动率: 输入市场价格反推隐含波动率')

    # 报告生成
    parser.add_argument('--report', action='store_true',
                        help='生成HTML报告 (需安装matplotlib)')

    # 真实市场数据
    parser.add_argument('--market-data', action='store_true',
                        help='使用akshare真实市场数据')

    # 亚式期权类型
    parser.add_argument('--asian-type', dest='asian_type', default='arithmetic',
                        choices=['arithmetic', 'geometric'],
                        help='亚式期权平均方式: arithmetic(算术), geometric(几何) [默认: arithmetic]')

    # 验证
    parser.add_argument('--verify', action='store_true',
                        help='运行Put-Call Parity验证和希腊字母数值验证')

    args = parser.parse_args()

    # 使用 rich 显示结果
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        from rich import box
        console = Console()
        use_rich = True
    except ImportError:
        use_rich = False
        console = None

    # 获取市场数据
    if args.market_data:
        try:
            from derivatives_pricer.data import get_realtime_params
            params = get_realtime_params()
            args.spot = params['spot']
            args.strike = params['strike']
            args.vol = params['vol']
            args.r = params['r']
            if use_rich:
                console.print(Panel(
                    f"[bold green]使用真实市场数据 (akshare)[/bold green]\n"
                    f"标的资产价格: {args.spot}\n"
                    f"行权价: {args.strike}\n"
                    f"波动率: {args.vol}\n"
                    f"无风险利率: {args.r}",
                    title="市场数据", border_style="green"
                ))
            else:
                print(f"=== 使用真实市场数据 (akshare) ===")
                print(f"  标的资产价格: {args.spot}")
                print(f"  行权价: {args.strike}")
                print(f"  波动率: {args.vol}")
                print(f"  无风险利率: {args.r}")
        except Exception as e:
            if use_rich:
                console.print(f"[yellow]无法获取真实数据: {e}, 使用默认参数[/yellow]")
            else:
                print(f"无法获取真实数据: {e}, 使用默认参数")

    # 参数验证
    if args.spot <= 0:
        print("错误: 标的资产价格必须为正")
        sys.exit(1)
    if args.strike <= 0:
        print("错误: 行权价必须为正")
        sys.exit(1)
    if args.vol <= 0:
        print("错误: 波动率必须为正")
        sys.exit(1)
    if args.t <= 0:
        print("错误: 到期时间必须为正")
        sys.exit(1)

    # 隐含波动率反推
    # 注意: implied_vol(price, S, K, T, r, option_type, q)
    if args.iv is not None:
        iv = implied_vol(args.iv, args.spot, args.strike, args.t, args.r,
                         option_type=args.option_type, q=args.q)
        if use_rich:
            console.print(Panel(
                f"[bold]市场价格: {args.iv:.6f}[/bold]\n"
                f"[bold green]隐含波动率: {iv:.6f} ({iv*100:.2f}%)[/bold green]",
                title="隐含波动率反推 (Newton-Raphson)", border_style="cyan"
            ))
        else:
            print(f"\n=== 隐含波动率反推 (Newton-Raphson) ===")
            print(f"  市场价格: {args.iv:.6f}")
            print(f"  隐含波动率: {iv:.6f} ({iv*100:.2f}%)")
        return

    # Put-Call Parity 验证
    # 注意: put_call_parity_check(S, K, T, r, sigma, q) 返回 dict
    if args.verify:
        parity = put_call_parity_check(args.spot, args.strike, args.t, args.r, args.vol, q=args.q)
        parity_ok = parity['difference'] < 1e-10
        error = parity['difference']
        if use_rich:
            status = "[green]通过[/green]" if parity_ok else "[red]失败[/red]"
            console.print(Panel(
                f"Put-Call Parity 验证: {status}\n"
                f"C - P = {parity['parity_left']:.6f}\n"
                f"S*e^(-qT) - K*e^(-rT) = {parity['parity_right']:.6f}\n"
                f"误差: {error:.2e}",
                title="验证", border_style="blue"
            ))
        else:
            print(f"\n=== Put-Call Parity 验证 ===")
            print(f"  结果: {'通过' if parity_ok else '失败'}")
            print(f"  C - P = {parity['parity_left']:.6f}")
            print(f"  S*e^(-qT) - K*e^(-rT) = {parity['parity_right']:.6f}")
            print(f"  误差: {error:.2e}")

    # 定价
    # 注意所有函数的参数顺序: (S/S0, K, T, r, sigma, option_type, q, ...)
    price = None
    method_name = ""
    extra_info = ""

    if args.opt_type == 'european':
        if args.method == 'bs':
            # bs_price(S, K, T, r, sigma, option_type, q)
            price = bs_price(args.spot, args.strike, args.t, args.r, args.vol,
                             option_type=args.option_type, q=args.q)
            method_name = "Black-Scholes 解析解"

        elif args.method == 'mc':
            # mc_antithetic(S0, K, T, r, sigma, option_type, q, n_paths) -> dict
            result = mc_antithetic(args.spot, args.strike, args.t, args.r, args.vol,
                                   option_type=args.option_type, q=args.q, n_paths=args.paths)
            price = result['price']
            se = result['std_error']
            method_name = "蒙特卡洛 (对偶变量法)"
            extra_info = f"路径数: {args.paths:,} | 标准误差: {se:.6f}"

        elif args.method == 'mc-cv':
            # mc_control_variate(S0, K, T, r, sigma, option_type, q, n_paths) -> dict
            result = mc_control_variate(args.spot, args.strike, args.t, args.r, args.vol,
                                        option_type=args.option_type, q=args.q, n_paths=args.paths)
            price = result['price']
            se = result['std_error']
            method_name = "蒙特卡洛 (控制变量法)"
            extra_info = f"路径数: {args.paths:,} | 标准误差: {se:.6f} | beta: {result.get('beta', 0):.4f}"

        elif args.method == 'fdm':
            # fdm_european(S0, K, T, r, sigma, option_type, q, method, N, M) -> dict
            result = fdm_european(args.spot, args.strike, args.t, args.r, args.vol,
                                  option_type=args.option_type, q=args.q,
                                  method='explicit', M=args.grid_m, N=args.grid_n)
            price = result['price']
            method_name = "有限差分法 (显式)"
            extra_info = f"网格: {args.grid_m}x{args.grid_n}"

        elif args.method == 'cn':
            # fdm_european with Crank-Nicolson
            result = fdm_european(args.spot, args.strike, args.t, args.r, args.vol,
                                  option_type=args.option_type, q=args.q,
                                  method='crank_nicolson', M=args.grid_m, N=args.grid_n)
            price = result['price']
            method_name = "有限差分法 (Crank-Nicolson)"
            extra_info = f"网格: {args.grid_m}x{args.grid_n}"

        elif args.method == 'cos':
            # cos_european(S0, K, r, q, sigma, T, option_type, N) -> float
            price = cos_european(args.spot, args.strike, args.r, args.q, args.vol, args.t,
                                 args.option_type, N=args.cos_n)
            method_name = "COS方法 (Fourier余弦展开)"
            extra_info = f"余弦项数: N={args.cos_n}"

    elif args.opt_type == 'american':
        if args.method == 'mc':
            # mc_american_ls(S0, K, T, r, sigma, option_type, q, n_paths, n_steps) -> dict
            result = mc_american_ls(args.spot, args.strike, args.t, args.r, args.vol,
                                    option_type=args.option_type, q=args.q,
                                    n_paths=args.paths, n_steps=50)
            price = result['price']
            method_name = "蒙特卡洛 (Longstaff-Schwartz)"
            extra_info = f"路径数: {args.paths:,} | 时间步数: 50"

        elif args.method in ('fdm', 'cn'):
            # fdm_american(S0, K, T, r, sigma, option_type, q, method, N, M) -> dict
            fdm_method = 'crank_nicolson' if args.method == 'cn' else 'explicit'
            result = fdm_american(args.spot, args.strike, args.t, args.r, args.vol,
                                  option_type=args.option_type, q=args.q,
                                  method=fdm_method, M=args.grid_m, N=args.grid_n)
            price = result['price']
            method_name = f"有限差分法 ({'Crank-Nicolson' if args.method == 'cn' else '显式'})"
            extra_info = f"网格: {args.grid_m}x{args.grid_n}"

        else:
            print(f"错误: 美式期权不支持方法 '{args.method}'，请使用 mc/fdm/cn")
            sys.exit(1)

    elif args.opt_type == 'asian':
        if args.method in ('mc', 'bs', 'cos'):
            # mc_asian(S0, K, T, r, sigma, option_type, q, avg_type, n_paths, n_steps) -> dict
            asian_avg = 'arithmetic' if args.asian_type == 'arithmetic' else 'geometric'
            result = mc_asian(args.spot, args.strike, args.t, args.r, args.vol,
                             option_type=args.option_type, q=args.q, avg_type=asian_avg,
                             n_paths=args.paths, n_steps=50)
            price = result['price']
            se = result['std_error']
            avg_label = '算术' if args.asian_type == 'arithmetic' else '几何'
            method_name = f"蒙特卡洛亚式期权 ({avg_label}平均)"
            extra_info = f"路径数: {args.paths:,} | 标准误差: {se:.6f}"

        else:
            # 对于亚式期权，默认使用蒙特卡洛
            asian_avg = 'arithmetic' if args.asian_type == 'arithmetic' else 'geometric'
            result = mc_asian(args.spot, args.strike, args.t, args.r, args.vol,
                             option_type=args.option_type, q=args.q, avg_type=asian_avg,
                             n_paths=args.paths, n_steps=50)
            price = result['price']
            se = result['std_error']
            avg_label = '算术' if args.asian_type == 'arithmetic' else '几何'
            method_name = f"蒙特卡洛亚式期权 ({avg_label}平均)"
            extra_info = f"路径数: {args.paths:,} | 标准误差: {se:.6f}"

    # 显示结果
    if price is not None:
        if use_rich:
            # 创建结果表格
            table = Table(title="定价结果", box=box.ROUNDED, show_header=True, header_style="bold magenta")
            table.add_column("项目", style="cyan", width=30)
            table.add_column("数值", style="green", width=30)

            table.add_row("期权类型", f"{args.opt_type.upper()} {args.option_type.upper()}")
            table.add_row("定价方法", method_name)
            table.add_row("标的资产价格 (S)", f"{args.spot:.4f}")
            table.add_row("行权价 (K)", f"{args.strike:.4f}")
            table.add_row("波动率 (sigma)", f"{args.vol:.4f} ({args.vol*100:.1f}%)")
            table.add_row("无风险利率 (r)", f"{args.r:.4f} ({args.r*100:.1f}%)")
            table.add_row("股息收益率 (q)", f"{args.q:.4f} ({args.q*100:.1f}%)")
            table.add_row("到期时间 (T)", f"{args.t:.4f} ({args.t*365:.0f}天)")
            table.add_row("[bold]期权价格[/bold]", f"[bold yellow]{price:.6f}[/bold yellow]")
            if extra_info:
                table.add_row("附加信息", extra_info)

            console.print(table)

            # 显示希腊字母 (仅欧式期权)
            if args.opt_type == 'european':
                # all_greeks(S, K, T, r, sigma, option_type, q)
                g = all_greeks(args.spot, args.strike, args.t, args.r, args.vol,
                               option_type=args.option_type, q=args.q)
                num_g = numerical_greeks(args.spot, args.strike, args.t, args.r, args.vol,
                                        option_type=args.option_type, q=args.q)

                greeks_table = Table(title="希腊字母", box=box.ROUNDED, show_header=True, header_style="bold blue")
                greeks_table.add_column("希腊字母", style="cyan", width=15)
                greeks_table.add_column("解析值", style="green", width=20)
                greeks_table.add_column("数值验证", style="yellow", width=20)
                greeks_table.add_column("误差", style="red", width=15)

                for name, val, num_val in [
                    ('Delta', g['delta'], num_g['delta']),
                    ('Gamma', g['gamma'], num_g['gamma']),
                    ('Theta', g['theta'], num_g['theta']),
                    ('Vega', g['vega'], num_g['vega']),
                    ('Rho', g['rho'], num_g['rho']),
                ]:
                    err = abs(val - num_val)
                    greeks_table.add_row(name, f"{val:.6f}", f"{num_val:.6f}", f"{err:.2e}")

                console.print(greeks_table)

        else:
            # 无 rich 的输出
            print(f"\n{'='*50}")
            print(f"  定价结果")
            print(f"{'='*50}")
            print(f"  期权类型:     {args.opt_type.upper()} {args.option_type.upper()}")
            print(f"  定价方法:     {method_name}")
            print(f"  标的价格 (S): {args.spot:.4f}")
            print(f"  行权价 (K):   {args.strike:.4f}")
            print(f"  波动率 (sigma): {args.vol:.4f} ({args.vol*100:.1f}%)")
            print(f"  无风险利率:   {args.r:.4f} ({args.r*100:.1f}%)")
            print(f"  股息收益率:   {args.q:.4f} ({args.q*100:.1f}%)")
            print(f"  到期时间 (T): {args.t:.4f} ({args.t*365:.0f}天)")
            print(f"{'='*50}")
            print(f"  >>> 期权价格: {price:.6f} <<<")
            if extra_info:
                print(f"  附加信息: {extra_info}")
            print(f"{'='*50}")

            if args.opt_type == 'european':
                g = all_greeks(args.spot, args.strike, args.t, args.r, args.vol,
                               option_type=args.option_type, q=args.q)
                num_g = numerical_greeks(args.spot, args.strike, args.t, args.r, args.vol,
                                        option_type=args.option_type, q=args.q)
                print(f"\n  希腊字母:")
                print(f"  {'名称':<10} {'解析值':>15} {'数值验证':>15} {'误差':>12}")
                print(f"  {'-'*55}")
                for name, val, num_val in [
                    ('Delta', g['delta'], num_g['delta']),
                    ('Gamma', g['gamma'], num_g['gamma']),
                    ('Theta', g['theta'], num_g['theta']),
                    ('Vega', g['vega'], num_g['vega']),
                    ('Rho', g['rho'], num_g['rho']),
                ]:
                    err = abs(val - num_val)
                    print(f"  {name:<10} {val:>15.6f} {num_val:>15.6f} {err:>12.2e}")

    # 生成HTML报告
    if args.report:
        try:
            from derivatives_pricer.report import save_html_report
            # save_html_report(S0, K, r, q, sigma, T, option_type, filepath)
            filepath = save_html_report(
                args.spot, args.strike, args.r, args.q, args.vol, args.t,
                args.option_type, filepath='option_report.html'
            )
            if use_rich:
                console.print(f"\n[bold green]HTML报告已生成: {filepath}[/bold green]")
            else:
                print(f"\nHTML报告已生成: {filepath}")
        except ImportError:
            print("警告: 生成报告需要 matplotlib，请安装: pip install matplotlib")
        except Exception as e:
            print(f"警告: 生成报告失败: {e}")


if __name__ == '__main__':
    main()
