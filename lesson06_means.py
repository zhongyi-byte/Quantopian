"""
第6讲：均值 (Means)
💡 Quantopian量化分析56讲

本教程介绍如何使用单一数值来概括数据集，核心目标是捕捉数据分布的关键信息。
"""

import yfinance as yf
import scipy.stats as stats
import numpy as np


def mode(l):
    """计算众数的自定义函数"""
    counts = {}
    for e in l:
        counts[e] = counts.get(e, 0) + 1

    maxcount = max(counts.values(), default=0)
    if maxcount < 1 and len(l) > 1:
        return 'No mode'

    modes = [k for k, v in counts.items() if v == maxcount]
    return modes if maxcount > 1 or len(l) == 1 else 'No mode'


def demonstrate_basic_means():
    """演示基本的均值计算"""
    print("="*50)
    print("基本均值计算示例")
    print("="*50)
    
    # 示例数据集
    x1 = [1, 2, 2, 3, 4, 5, 5, 7]
    x2 = x1 + [100]
    
    print(f"数据集 x1: {x1}")
    print(f"数据集 x2: {x2}")
    print()
    
    # 算术平均数
    print("1. 算术平均数 (Arithmetic Mean)")
    print(f"x1的平均数: {np.mean(x1):.2f}")
    print(f"x2的平均数: {np.mean(x2):.2f}")
    print("注意：平均数容易受异常值影响")
    print()
    
    # 中位数
    print("2. 中位数 (Median)")
    print(f"x1的中位数: {np.median(x1)}")
    print(f"x2的中位数: {np.median(x2)}")
    print("注意：中位数对异常值不敏感")
    print()
    
    # 众数
    print("3. 众数 (Mode)")
    print(f"x1的众数: {mode(x1)}")
    print(f"x2的众数: {mode(x2)}")
    print()
    
    # 几何平均数
    print("4. 几何平均数 (Geometric Mean)")
    print(f"x1的几何平均数: {stats.gmean(x1):.2f}")
    print(f"x2的几何平均数: {stats.gmean(x2):.2f}")
    print()
    
    # 调和平均数
    print("5. 调和平均数 (Harmonic Mean)")
    print(f"x1的调和平均数: {stats.hmean(x1):.2f}")
    print(f"x2的调和平均数: {stats.hmean(x2):.2f}")
    print()


def analyze_financial_data():
    """分析实际金融数据"""
    print("="*50)
    print("金融数据分析示例")
    print("="*50)
    
    # 获取标普500指数数据
    start = '2014-01-01'
    end = '2015-01-01'
    
    print(f"获取SPY数据: {start} 到 {end}")
    data = yf.download('SPY', start=start, end=end, progress=False)
    prices = data['Close']
    
    # 计算日收益率
    returns = prices.pct_change().dropna()
    
    print(f"价格数据点数: {len(prices)}")
    print(f"收益率数据点数: {len(returns)}")
    print()
    
    # 展示基本统计量
    print("收益率统计分析:")
    print(f"收益率均值: {returns.mean():.6f}")
    print(f"收益率中位数: {returns.median():.6f}")
    
    # 计算众数（需要分组）
    returns_rounded = returns.round(4)
    returns_mode = mode(returns_rounded.tolist())
    print(f"收益率众数 (四舍五入到4位): {returns_mode}")
    
    # 处理调和平均数（收益率可能为负，需要特殊处理）
    if (returns > 0).all():
        harmonic_mean_return = stats.hmean(returns)
        print(f"收益率调和平均数: {harmonic_mean_return:.6f}")
    else:
        print("收益率调和平均数: 无法计算（存在负值或零值）")
    print()
    
    # 几何平均收益率计算
    geo_mean_return = stats.gmean(returns + 1) - 1
    print(f"几何平均收益率: {geo_mean_return:.6f}")
    print()
    
    # 验证几何平均计算
    print("几何平均计算验证:")
    initial_price = prices.iloc[0]
    final_price = prices.iloc[-1]
    T = len(returns)
    calculated_price = initial_price * (1 + geo_mean_return)**T
    
    print(f"初始价格: ${initial_price:.2f}")
    print(f"实际最终价格: ${final_price:.2f}")
    print(f"几何平均计算价格: ${calculated_price:.2f}")
    print(f"误差: ${abs(final_price - calculated_price):.2f}")
    print()


def multi_stock_analysis():
    """多股票分析示例"""
    print("="*50)
    print("多股票分析示例")
    print("="*50)
    
    # 多股票分析
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    start = '2014-01-01'
    end = '2015-01-01'
    
    print(f"分析股票: {', '.join(tickers)}")
    multi_data = yf.download(tickers, start=start, end=end, progress=False)['Close']
    
    # 计算各股平均收益率
    returns_multi = multi_data.pct_change().dropna()
    avg_returns = returns_multi.mean()
    
    print("\n各股票平均日收益率:")
    for ticker in tickers:
        print(f"{ticker}: {avg_returns[ticker]:.6f}")
    
    # 计算几何平均收益率
    print("\n各股票几何平均日收益率:")
    for ticker in tickers:
        geo_return = stats.gmean(returns_multi[ticker] + 1) - 1
        print(f"{ticker}: {geo_return:.6f}")
    print()


def demonstrate_mean_properties():
    """演示不同均值的特性"""
    print("="*50)
    print("不同均值的特性比较")
    print("="*50)
    
    # 正态分布数据
    np.random.seed(42)
    normal_data = np.random.normal(100, 15, 1000)
    
    # 偏态分布数据（对数正态分布）
    skewed_data = np.random.lognormal(4, 0.5, 1000)
    
    datasets = {
        "正态分布数据": normal_data,
        "偏态分布数据": skewed_data
    }
    
    for name, data in datasets.items():
        print(f"\n{name}:")
        print(f"  算术平均数: {np.mean(data):.2f}")
        print(f"  中位数: {np.median(data):.2f}")
        print(f"  几何平均数: {stats.gmean(data):.2f}")
        print(f"  调和平均数: {stats.hmean(data):.2f}")


def main():
    """主函数"""
    print("第6讲：均值 (Means)")
    print("Quantopian量化分析56讲\n")
    
    # 基本均值计算
    demonstrate_basic_means()
    
    # 金融数据分析
    try:
        analyze_financial_data()
        multi_stock_analysis()
    except Exception as e:
        print(f"获取金融数据时出错: {e}")
        print("请检查网络连接或稍后再试")
    
    # 均值特性比较
    demonstrate_mean_properties()
    
    print("="*50)
    print("关键要点:")
    print("1. 算术平均数容易受极端值影响")
    print("2. 中位数对偏态数据更具代表性")
    print("3. 几何平均数适用于增长率计算")
    print("4. 调和平均数适用于比率数据")
    print("5. 结合多个统计量进行综合分析")
    print("="*50)


if __name__ == "__main__":
    main()