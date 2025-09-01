"""
第7讲：方差 (Variance) - 简化版本
💡 Quantopian量化分析56讲

核心概念演示：极差、平均绝对偏差、方差与标准差、半方差
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings

# 设置随机种子保证结果可复现
np.random.seed(121)

# 忽略警告信息
warnings.filterwarnings('ignore')


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
    print(f"   最大值: {X.max()}, 最小值: {X.min()}")
    print()
    
    # 2. 平均绝对偏差（Mean Absolute Deviation）
    mean_val = np.mean(X)
    abs_deviations = np.abs(X - mean_val)
    mad = np.mean(abs_deviations)
    print(f"2. 平均绝对偏差（MAD）: {mad:.2f}")
    print(f"   数据均值: {mean_val:.2f}")
    print(f"   公式: MAD = (1/n) * Σ|xi - μ|")
    print()
    
    # 3. 方差和标准差
    variance = np.var(X, ddof=1)  # 样本方差
    std_dev = np.std(X, ddof=1)   # 样本标准差
    print(f"3. 样本方差: {variance:.2f}")
    print(f"   样本标准差: {std_dev:.2f}")
    print(f"   变异系数: {std_dev/mean_val:.4f}")
    print()
    
    # 4. 半方差（下行风险）
    downside_data = X[X <= mean_val]
    if len(downside_data) > 0:
        semivariance = np.sum((downside_data - mean_val)**2) / len(downside_data)
        semi_std = np.sqrt(semivariance)
        print(f"4. 半方差: {semivariance:.2f}")
        print(f"   半标准差: {semi_std:.2f}")
        print(f"   低于均值的数据点: {len(downside_data)}/{len(X)}")
    print()
    
    return X, mean_val, std_dev


def simulate_stock_returns():
    """模拟股票收益率分析"""
    print("=" * 60)
    print("模拟股票收益率风险分析")
    print("=" * 60)
    
    # 模拟250个交易日的收益率数据
    np.random.seed(42)
    daily_returns = np.random.normal(0.001, 0.02, 250)  # 日均收益0.1%，日波动2%
    
    # 计算风险指标
    mean_return = np.mean(daily_returns)
    annual_return = mean_return * 252
    daily_vol = np.std(daily_returns, ddof=1)
    annual_vol = daily_vol * np.sqrt(252)
    
    print(f"模拟股票收益率统计:")
    print(f"日均收益率: {mean_return:.4f} ({mean_return*100:.2f}%)")
    print(f"年化收益率: {annual_return:.4f} ({annual_return*100:.2f}%)")
    print(f"日波动率: {daily_vol:.4f} ({daily_vol*100:.2f}%)")
    print(f"年化波动率: {annual_vol:.4f} ({annual_vol*100:.2f}%)")
    print()
    
    # 下行风险分析
    downside_returns = daily_returns[daily_returns < mean_return]
    if len(downside_returns) > 0:
        downside_deviation = np.sqrt(np.mean((downside_returns - mean_return)**2))
        annual_downside_vol = downside_deviation * np.sqrt(252)
        
        print(f"下行风险分析:")
        print(f"下行标准差: {downside_deviation:.4f}")
        print(f"年化下行波动率: {annual_downside_vol:.2%}")
        print(f"负收益天数: {len(downside_returns)}/250")
    
    # VaR计算
    var_5 = np.percentile(daily_returns, 5)
    var_1 = np.percentile(daily_returns, 1)
    print(f"\n风险价值（VaR）:")
    print(f"5% VaR: {var_5:.4f} ({var_5*100:.2f}%)")
    print(f"1% VaR: {var_1:.4f} ({var_1*100:.2f}%)")
    
    return daily_returns


def visualize_risk_concepts():
    """可视化风险概念"""
    print("\n" + "=" * 60)
    print("风险概念可视化")
    print("=" * 60)
    
    # 生成三种不同风险特征的数据
    np.random.seed(123)
    
    low_risk = np.random.normal(0.05, 0.1, 1000)      # 低风险：小波动
    medium_risk = np.random.normal(0.08, 0.2, 1000)   # 中等风险：中等波动
    high_risk = np.random.normal(0.12, 0.35, 1000)    # 高风险：大波动
    
    datasets = {
        '低风险资产': low_risk,
        '中风险资产': medium_risk, 
        '高风险资产': high_risk
    }
    
    colors = ['green', 'orange', 'red']
    
    # 创建图表
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 收益率分布对比
    for i, (name, data) in enumerate(datasets.items()):
        ax1.hist(data, bins=50, alpha=0.6, label=name, color=colors[i], density=True)
        ax1.axvline(np.mean(data), color=colors[i], linestyle='--', alpha=0.8)
    
    ax1.set_title('不同风险资产收益率分布')
    ax1.set_xlabel('收益率')
    ax1.set_ylabel('密度')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 风险指标对比
    risk_metrics = {}
    asset_names = []
    volatilities = []
    sharpe_ratios = []
    
    for name, data in datasets.items():
        vol = np.std(data, ddof=1)
        ret = np.mean(data)
        sharpe = ret / vol if vol > 0 else 0
        
        asset_names.append(name)
        volatilities.append(vol)
        sharpe_ratios.append(sharpe)
    
    x_pos = np.arange(len(asset_names))
    ax2.bar(x_pos, volatilities, color=colors)
    ax2.set_title('波动率对比')
    ax2.set_xlabel('资产类型')
    ax2.set_ylabel('波动率')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(asset_names, rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # 3. 累计收益模拟
    for i, (name, data) in enumerate(datasets.items()):
        cumulative = np.cumprod(1 + data[:252])  # 一年的数据
        ax3.plot(cumulative, label=name, color=colors[i])
    
    ax3.set_title('累计收益对比（1年期）')
    ax3.set_xlabel('交易日')
    ax3.set_ylabel('累计收益倍数')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 风险-收益散点图
    returns = [np.mean(data) for data in datasets.values()]
    risks = [np.std(data, ddof=1) for data in datasets.values()]
    
    ax4.scatter(risks, returns, c=colors, s=100, alpha=0.7)
    for i, name in enumerate(asset_names):
        ax4.annotate(name, (risks[i], returns[i]), 
                    xytext=(5, 5), textcoords='offset points')
    
    ax4.set_title('风险-收益关系')
    ax4.set_xlabel('风险（标准差）')
    ax4.set_ylabel('期望收益')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 输出数值结果
    print("\n风险指标对比:")
    print("-" * 40)
    for i, name in enumerate(asset_names):
        print(f"{name}:")
        print(f"  期望收益: {returns[i]:.3f} ({returns[i]*100:.1f}%)")
        print(f"  风险(波动率): {risks[i]:.3f} ({risks[i]*100:.1f}%)")
        print(f"  夏普比率: {sharpe_ratios[i]:.2f}")
        print()


def demonstrate_target_semivariance():
    """演示目标半方差的概念"""
    print("=" * 60)
    print("目标半方差分析")
    print("=" * 60)
    
    # 生成收益率数据
    np.random.seed(456)
    returns = np.random.normal(0.08, 0.25, 500)  # 年化8%收益，25%波动
    
    # 不同的目标收益率
    targets = [0.0, 0.05, 0.1, 0.15]
    
    print("不同目标收益率的半方差分析:")
    print("-" * 40)
    
    for target in targets:
        # 计算目标半方差
        downside_returns = returns[returns < target]
        
        if len(downside_returns) > 0:
            target_semivar = np.mean((downside_returns - target)**2)
            target_semi_std = np.sqrt(target_semivar)
            downside_prob = len(downside_returns) / len(returns)
            
            print(f"目标收益率 {target:.1%}:")
            print(f"  目标半方差: {target_semivar:.6f}")
            print(f"  目标半标准差: {target_semi_std:.4f}")
            print(f"  低于目标概率: {downside_prob:.2%}")
            print()
        else:
            print(f"目标收益率 {target:.1%}: 无数据低于目标")
            print()


def main():
    """主函数"""
    print("第7讲：方差 (Variance)")
    print("Quantopian量化分析56讲 - 简化演示版")
    print()
    
    # 1. 基本概念演示
    X, mean_val, std_dev = demonstrate_basic_variance_concepts()
    
    # 2. 股票收益率模拟分析
    daily_returns = simulate_stock_returns()
    
    # 3. 可视化风险概念
    visualize_risk_concepts()
    
    # 4. 目标半方差演示
    demonstrate_target_semivariance()
    
    # 总结要点
    print("=" * 60)
    print("关键要点总结:")
    print("=" * 60)
    print("1. 极差：反映数据的最大波动范围，但对异常值敏感")
    print("2. 平均绝对偏差：衡量数据与均值的平均距离")
    print("3. 方差/标准差：量化波动性的核心指标，广泛应用于风险管理")
    print("4. 半方差/半标准差：只关注下行风险，更符合投资者风险厌恶特征")
    print("5. 目标半方差：相对特定目标的下行风险度量")
    print("6. 年化处理：金融数据需要标准化到年度基础进行比较")
    print("7. VaR：量化极端损失的概率性风险指标")
    print("=" * 60)


if __name__ == "__main__":
    main()