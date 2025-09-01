"""
第7讲：方差 (Variance) - 文本输出版
💡 Quantopian量化分析56讲

核心概念演示：极差、平均绝对偏差、方差与标准差、半方差
"""

import numpy as np

# 设置随机种子保证结果可复现
np.random.seed(121)


def demonstrate_basic_variance_concepts():
    """演示方差的基本概念"""
    print("=" * 60)
    print("方差与离散度量基本概念演示")
    print("=" * 60)
    
    # 创建示例数据
    X = np.sort(np.random.randint(100, size=20))
    print("示例数据集 X:")
    print(X)
    print()
    
    # 1. 极差（Range）
    data_range = np.ptp(X)  # peak to peak
    print(f"1. 极差（Range）: {data_range}")
    print(f"   公式: Max - Min = {X.max()} - {X.min()} = {data_range}")
    print(f"   含义: 衡量数据分布的最大波动范围")
    print()
    
    # 2. 平均绝对偏差（Mean Absolute Deviation）
    mean_val = np.mean(X)
    abs_deviations = np.abs(X - mean_val)
    mad = np.mean(abs_deviations)
    print(f"2. 平均绝对偏差（MAD）: {mad:.2f}")
    print(f"   数据均值: {mean_val:.2f}")
    print(f"   公式: MAD = (1/n) * Σ|xi - μ|")
    print(f"   含义: 数据点与均值的平均距离")
    print()
    
    # 3. 方差和标准差
    # 总体方差和样本方差
    pop_variance = np.var(X, ddof=0)      # 总体方差（除以n）
    sample_variance = np.var(X, ddof=1)   # 样本方差（除以n-1）
    pop_std = np.std(X, ddof=0)          # 总体标准差
    sample_std = np.std(X, ddof=1)       # 样本标准差
    
    print(f"3. 方差和标准差:")
    print(f"   总体方差: {pop_variance:.2f}")
    print(f"   样本方差: {sample_variance:.2f}")
    print(f"   总体标准差: {pop_std:.2f}")
    print(f"   样本标准差: {sample_std:.2f}")
    print(f"   变异系数: {sample_std/mean_val:.4f}")
    print(f"   含义: 量化数据相对于均值的离散程度")
    print()
    
    # 4. 半方差（下行风险）
    downside_data = X[X <= mean_val]
    upside_data = X[X > mean_val]
    
    if len(downside_data) > 0:
        semivariance = np.sum((downside_data - mean_val)**2) / len(downside_data)
        semi_std = np.sqrt(semivariance)
        print(f"4. 半方差和半标准差（下行风险）:")
        print(f"   半方差: {semivariance:.2f}")
        print(f"   半标准差: {semi_std:.2f}")
        print(f"   低于均值的数据点: {len(downside_data)}/{len(X)} ({len(downside_data)/len(X)*100:.1f}%)")
        print(f"   高于均值的数据点: {len(upside_data)}/{len(X)} ({len(upside_data)/len(X)*100:.1f}%)")
        print(f"   含义: 只考虑负偏差的风险度量")
    print()
    
    # 5. 目标半方差示例
    target = 50  # 设定目标值
    target_downside = X[X < target]
    
    if len(target_downside) > 0:
        target_semivar = np.mean((target_downside - target)**2)
        target_semi_std = np.sqrt(target_semivar)
        print(f"5. 目标半方差（目标值={target}）:")
        print(f"   目标半方差: {target_semivar:.2f}")
        print(f"   目标半标准差: {target_semi_std:.2f}")
        print(f"   低于目标的数据点: {len(target_downside)}/{len(X)} ({len(target_downside)/len(X)*100:.1f}%)")
        print(f"   含义: 相对于特定目标值的下行风险")
    else:
        print(f"5. 目标半方差（目标值={target}）:")
        print(f"   无数据点低于目标值")
    print()
    
    return X, mean_val, sample_std


def simulate_financial_scenarios():
    """模拟金融场景分析"""
    print("=" * 60)
    print("金融风险分析场景模拟")
    print("=" * 60)
    
    # 模拟三种不同的投资产品
    scenarios = {
        '货币基金': {'mean': 0.0002, 'std': 0.0001, 'days': 252},  # 年化2%，极低波动
        '债券基金': {'mean': 0.0003, 'std': 0.005, 'days': 252},   # 年化3%，低波动
        '股票基金': {'mean': 0.0005, 'std': 0.02, 'days': 252},    # 年化5%，高波动
        '加密货币': {'mean': 0.001, 'std': 0.05, 'days': 252}      # 年化10%，极高波动
    }
    
    print("不同投资产品风险特征对比:")
    print("-" * 80)
    print(f"{'产品':<10} {'日均收益':<10} {'年化收益':<10} {'日波动率':<10} {'年化波动':<10} {'夏普比率':<10}")
    print("-" * 80)
    
    for name, params in scenarios.items():
        # 模拟收益率数据
        np.random.seed(42)  # 固定种子确保可重复
        daily_returns = np.random.normal(params['mean'], params['std'], params['days'])
        
        # 计算风险指标
        daily_mean = np.mean(daily_returns)
        annual_return = daily_mean * 252
        daily_vol = np.std(daily_returns, ddof=1)
        annual_vol = daily_vol * np.sqrt(252)
        sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
        
        print(f"{name:<10} {daily_mean:<10.4f} {annual_return:<10.2%} {daily_vol:<10.4f} {annual_vol:<10.2%} {sharpe_ratio:<10.2f}")
    
    print("-" * 80)
    print()
    
    # 详细分析股票基金
    print("股票基金详细风险分析:")
    print("-" * 40)
    
    np.random.seed(123)
    stock_returns = np.random.normal(0.0005, 0.02, 252)
    
    # 基础统计
    mean_return = np.mean(stock_returns)
    annual_return = mean_return * 252
    daily_vol = np.std(stock_returns, ddof=1)
    annual_vol = daily_vol * np.sqrt(252)
    
    print(f"基础统计指标:")
    print(f"  日均收益率: {mean_return:.4f} ({mean_return*100:.2f}%)")
    print(f"  年化收益率: {annual_return:.2%}")
    print(f"  日波动率: {daily_vol:.4f} ({daily_vol*100:.2f}%)")
    print(f"  年化波动率: {annual_vol:.2%}")
    print()
    
    # 风险分析
    negative_returns = stock_returns[stock_returns < 0]
    positive_returns = stock_returns[stock_returns >= 0]
    
    print(f"收益率分布:")
    print(f"  正收益天数: {len(positive_returns)} ({len(positive_returns)/252*100:.1f}%)")
    print(f"  负收益天数: {len(negative_returns)} ({len(negative_returns)/252*100:.1f}%)")
    print(f"  最大单日收益: {np.max(stock_returns):.2%}")
    print(f"  最大单日亏损: {np.min(stock_returns):.2%}")
    print()
    
    # 下行风险分析
    downside_returns = stock_returns[stock_returns < mean_return]
    if len(downside_returns) > 0:
        downside_deviation = np.sqrt(np.mean((downside_returns - mean_return)**2))
        annual_downside_vol = downside_deviation * np.sqrt(252)
        sortino_ratio = annual_return / annual_downside_vol
        
        print(f"下行风险指标:")
        print(f"  下行标准差: {downside_deviation:.4f}")
        print(f"  年化下行波动率: {annual_downside_vol:.2%}")
        print(f"  索蒂诺比率: {sortino_ratio:.2f}")
        print()
    
    # VaR分析
    var_levels = [1, 5, 10]
    print(f"风险价值（VaR）分析:")
    for level in var_levels:
        var_value = np.percentile(stock_returns, level)
        print(f"  {level}% VaR: {var_value:.2%} (日)")
        
        # 条件VaR (CVaR)
        cvar_returns = stock_returns[stock_returns <= var_value]
        if len(cvar_returns) > 0:
            cvar_value = np.mean(cvar_returns)
            print(f"  {level}% CVaR: {cvar_value:.2%} (日)")
    print()


def compare_risk_measures():
    """比较不同风险度量方法"""
    print("=" * 60)
    print("不同风险度量方法的比较")
    print("=" * 60)
    
    # 创建三种不同分布的数据
    np.random.seed(456)
    
    # 1. 正态分布数据
    normal_data = np.random.normal(0.05, 0.2, 1000)
    
    # 2. 偏态分布数据（更多负收益）
    skewed_data = np.concatenate([
        np.random.normal(-0.1, 0.1, 400),  # 40%的负收益期
        np.random.normal(0.15, 0.15, 600)  # 60%的正收益期
    ])
    
    # 3. 厚尾分布数据（极端值更多）
    fat_tail_data = np.concatenate([
        np.random.normal(0.05, 0.1, 900),  # 90%正常波动
        np.random.normal(0, 0.8, 100)      # 10%极端波动
    ])
    
    datasets = {
        '正态分布': normal_data,
        '偏态分布': skewed_data,
        '厚尾分布': fat_tail_data
    }
    
    print("三种分布的风险度量对比:")
    print("-" * 90)
    print(f"{'分布类型':<12} {'均值':<8} {'标准差':<8} {'偏度':<8} {'峰度':<8} {'5%VaR':<8} {'最小值':<8}")
    print("-" * 90)
    
    for name, data in datasets.items():
        # 计算各种统计量
        mean_val = np.mean(data)
        std_val = np.std(data, ddof=1)
        
        # 偏度和峰度（简化计算）
        centered = data - mean_val
        skewness = np.mean(centered**3) / (std_val**3)
        kurtosis = np.mean(centered**4) / (std_val**4) - 3  # 超额峰度
        
        var_5 = np.percentile(data, 5)
        min_val = np.min(data)
        
        print(f"{name:<12} {mean_val:<8.3f} {std_val:<8.3f} {skewness:<8.2f} {kurtosis:<8.2f} {var_5:<8.3f} {min_val:<8.3f}")
    
    print("-" * 90)
    print()
    
    # 详细分析每种分布
    for name, data in datasets.items():
        print(f"{name}详细分析:")
        print("-" * 30)
        
        mean_val = np.mean(data)
        std_val = np.std(data, ddof=1)
        
        # 不同风险度量
        mad = np.mean(np.abs(data - mean_val))
        range_val = np.ptp(data)
        
        # 半方差
        downside_data = data[data < mean_val]
        if len(downside_data) > 0:
            semivar = np.mean((downside_data - mean_val)**2)
            semi_std = np.sqrt(semivar)
        else:
            semi_std = 0
        
        # 不同百分位数的VaR
        var_1 = np.percentile(data, 1)
        var_5 = np.percentile(data, 5)
        var_10 = np.percentile(data, 10)
        
        print(f"  均值: {mean_val:.4f}")
        print(f"  标准差: {std_val:.4f}")
        print(f"  平均绝对偏差: {mad:.4f}")
        print(f"  极差: {range_val:.4f}")
        print(f"  半标准差: {semi_std:.4f}")
        print(f"  1% VaR: {var_1:.4f}")
        print(f"  5% VaR: {var_5:.4f}")
        print(f"  10% VaR: {var_10:.4f}")
        print()


def practical_applications():
    """实际应用示例"""
    print("=" * 60)
    print("实际投资组合风险管理应用")
    print("=" * 60)
    
    # 模拟投资组合
    np.random.seed(789)
    
    # 三个资产的收益率
    asset_returns = {
        '股票': np.random.normal(0.08, 0.25, 252),    # 8%年收益，25%波动
        '债券': np.random.normal(0.04, 0.08, 252),    # 4%年收益，8%波动
        '商品': np.random.normal(0.06, 0.30, 252)     # 6%年收益，30%波动
    }
    
    # 不同的投资组合权重
    portfolios = {
        '保守组合': [0.2, 0.7, 0.1],   # 20%股票, 70%债券, 10%商品
        '平衡组合': [0.5, 0.4, 0.1],   # 50%股票, 40%债券, 10%商品
        '激进组合': [0.7, 0.2, 0.1]    # 70%股票, 20%债券, 10%商品
    }
    
    print("投资组合风险分析:")
    print("-" * 60)
    
    for portfolio_name, weights in portfolios.items():
        # 计算组合收益率
        portfolio_returns = (
            weights[0] * asset_returns['股票'] +
            weights[1] * asset_returns['债券'] +
            weights[2] * asset_returns['商品']
        )
        
        # 风险指标
        portfolio_mean = np.mean(portfolio_returns)
        portfolio_std = np.std(portfolio_returns, ddof=1)
        sharpe_ratio = portfolio_mean / portfolio_std if portfolio_std > 0 else 0
        
        # 下行风险
        downside_returns = portfolio_returns[portfolio_returns < portfolio_mean]
        if len(downside_returns) > 0:
            downside_dev = np.sqrt(np.mean((downside_returns - portfolio_mean)**2))
            sortino_ratio = portfolio_mean / downside_dev
        else:
            sortino_ratio = 0
        
        # VaR
        var_5 = np.percentile(portfolio_returns, 5)
        
        # 最大回撤模拟
        cumulative_returns = np.cumprod(1 + portfolio_returns)
        rolling_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = np.min(drawdowns)
        
        print(f"{portfolio_name} (权重: {weights}):")
        print(f"  年化收益率: {portfolio_mean * 252:.2%}")
        print(f"  年化波动率: {portfolio_std * np.sqrt(252):.2%}")
        print(f"  夏普比率: {sharpe_ratio * np.sqrt(252):.2f}")
        print(f"  索蒂诺比率: {sortino_ratio * np.sqrt(252):.2f}")
        print(f"  5% VaR (日): {var_5:.2%}")
        print(f"  最大回撤: {max_drawdown:.2%}")
        print()


def main():
    """主函数"""
    print("第7讲：方差 (Variance)")
    print("Quantopian量化分析56讲 - 文本演示版")
    print()
    
    # 1. 基本概念演示
    X, mean_val, std_dev = demonstrate_basic_variance_concepts()
    
    # 2. 金融场景模拟
    simulate_financial_scenarios()
    
    # 3. 风险度量方法比较
    compare_risk_measures()
    
    # 4. 实际应用示例
    practical_applications()
    
    # 总结要点
    print("=" * 60)
    print("关键要点总结:")
    print("=" * 60)
    print("1. 极差：简单但对异常值敏感，适合快速了解数据范围")
    print("2. 平均绝对偏差：相比标准差对异常值不那么敏感")
    print("3. 标准差：最常用的风险度量，便于数学处理和比较")
    print("4. 半标准差：专注下行风险，更符合投资者心理")
    print("5. VaR：量化极端损失的概率度量，监管要求")
    print("6. 目标半方差：相对特定基准的风险度量")
    print("7. 组合效应：分散化可以降低总体风险")
    print("8. 时间维度：需要考虑持有期间和频率调整")
    print("=" * 60)


if __name__ == "__main__":
    main()