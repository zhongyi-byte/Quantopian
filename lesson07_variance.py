"""
第7讲：方差 (Variance)
💡 Quantopian量化分析56讲

离散度量用于量化数据分布的波动程度，在金融领域直接关联风险度量。
本教程涵盖：极差、平均绝对偏差、方差与标准差、半方差与半标准差、目标半方差。
"""

import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import warnings

# 设置随机种子保证结果可复现
np.random.seed(121)

# 忽略yfinance的警告信息
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def demonstrate_range():
    """演示极差（Range）的计算"""
    print("=" * 60)
    print("1. 极差（Range）- 衡量数据分布的最大波动范围")
    print("=" * 60)
    
    # 随机数据示例
    X = np.sort(np.random.randint(100, size=20))
    print("随机数据集X:")
    print(X)
    print(f"极差: {np.ptp(X)}")
    print(f"最大值: {X.max()}, 最小值: {X.min()}")
    print()
    
    return X


def analyze_stock_range(symbol='AAPL', start='2022-01-01', end='2023-01-01'):
    """分析股票价格的极差"""
    print(f"{symbol}股价极差分析 ({start} 到 {end}):")
    
    try:
        # 获取股价数据
        stock_data = yf.download(symbol, start=start, end=end, progress=False)['Close']
        
        price_range = stock_data.max() - stock_data.min()
        max_date = str(stock_data.idxmax().date())
        min_date = str(stock_data.idxmin().date())
        
        print(f"股价极差: ${price_range:.2f}")
        print(f"最高价: ${stock_data.max():.2f} (日期: {max_date})")
        print(f"最低价: ${stock_data.min():.2f} (日期: {min_date})")
        print(f"相对波动幅度: {float(price_range/stock_data.mean()):.2%}")
        
        return stock_data
        
    except Exception as e:
        print(f"获取{symbol}数据时出错: {e}")
        return None


def calculate_mad(data):
    """计算平均绝对偏差（Mean Absolute Deviation）"""
    print("=" * 60)
    print("2. 平均绝对偏差（MAD）- 衡量数据与均值的平均偏离程度")
    print("=" * 60)
    
    mu = np.mean(data)
    abs_deviations = np.abs(data - mu)
    mad = np.mean(abs_deviations)
    
    print(f"数据均值: {mu:.2f}")
    print(f"平均绝对偏差: {mad:.2f}")
    print(f"公式: MAD = (1/n) * Σ|xi - μ|")
    print()
    
    return mad


def analyze_variance_std(data):
    """分析方差和标准差"""
    print("=" * 60)
    print("3. 方差与标准差 - 量化数据波动性的核心指标")
    print("=" * 60)
    
    # 使用样本方差和标准差（除以n-1）
    variance = np.var(data, ddof=1)
    std_dev = np.std(data, ddof=1)
    
    print(f"样本方差: {variance:.2f}")
    print(f"样本标准差: {std_dev:.2f}")
    print(f"变异系数: {std_dev/np.mean(data):.4f}")
    print()
    
    return variance, std_dev


def analyze_stock_volatility(symbol='TSLA', start='2022-01-01', end='2023-01-01'):
    """分析股票收益率波动"""
    print(f"{symbol}收益率波动分析:")
    
    try:
        # 获取股价数据
        stock_data = yf.download(symbol, start=start, end=end, progress=False)['Close']
        returns = stock_data.pct_change().dropna()
        
        # 计算波动率指标
        daily_vol = returns.std()
        annual_vol = daily_vol * np.sqrt(252)  # 年化波动率
        
        print(f"日收益率标准差: {daily_vol:.4f}")
        print(f"年化波动率: {float(annual_vol):.2%}")
        print(f"最大单日收益: {float(returns.max()):.2%}")
        print(f"最大单日亏损: {float(returns.min()):.2%}")
        print()
        
        # 可视化收益率分布
        plt.figure(figsize=(12, 8))
        
        # 累计收益走势
        plt.subplot(2, 2, 1)
        cumulative_returns = (1 + returns).cumprod()
        plt.plot(cumulative_returns.index, cumulative_returns.values)
        plt.title(f'{symbol} 累计收益走势')
        plt.ylabel('累计收益倍数')
        
        # 收益率时间序列
        plt.subplot(2, 2, 2)
        plt.plot(returns.index, returns.values, alpha=0.7)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        plt.title(f'{symbol} 日收益率时间序列')
        plt.ylabel('日收益率')
        
        # 收益率直方图
        plt.subplot(2, 2, 3)
        plt.hist(returns, bins=50, alpha=0.7, density=True)
        plt.axvline(returns.mean(), color='red', linestyle='--', label='均值')
        plt.axvline(returns.mean() + returns.std(), color='orange', linestyle='--', label='+1σ')
        plt.axvline(returns.mean() - returns.std(), color='orange', linestyle='--', label='-1σ')
        plt.title('收益率分布')
        plt.xlabel('日收益率')
        plt.ylabel('密度')
        plt.legend()
        
        # Q-Q图检验正态性
        try:
            from scipy import stats
            plt.subplot(2, 2, 4)
            stats.probplot(returns, dist="norm", plot=plt)
            plt.title('Q-Q图 (正态性检验)')
            plt.grid(True, alpha=0.3)
        except ImportError:
            plt.subplot(2, 2, 4)
            plt.text(0.5, 0.5, '需要安装scipy包\n用于Q-Q图分析', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Q-Q图 (需要scipy)')
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return returns
        
    except Exception as e:
        print(f"获取{symbol}数据时出错: {e}")
        return None


def calculate_semivariance(data, target=None):
    """计算半方差和半标准差"""
    print("=" * 60)
    print("4. 半方差与半标准差 - 专门衡量下行风险的指标")
    print("=" * 60)
    
    if target is None:
        target = np.mean(data)
    
    # 计算低于目标值的数据点
    downside_data = data[data <= target]
    
    if len(downside_data) > 0:
        # 半方差：只考虑低于目标值的偏差
        semivariance = np.sum((downside_data - target)**2) / len(downside_data)
        semi_std = np.sqrt(semivariance)
        
        print(f"目标值: {target:.4f}")
        print(f"低于目标值的数据点: {len(downside_data)}/{len(data)}")
        print(f"半方差: {semivariance:.6f}")
        print(f"半标准差: {semi_std:.6f}")
        print(f"下行偏差占比: {len(downside_data)/len(data):.2%}")
        
        return semivariance, semi_std
    else:
        print("无数据点低于目标值")
        return 0, 0


def analyze_downside_risk(symbol='BTC-USD', start='2021-01-01', end='2023-01-01'):
    """分析加密货币的下行风险"""
    print(f"{symbol}下行风险分析:")
    
    try:
        # 获取价格数据
        crypto_data = yf.download(symbol, start=start, end=end, progress=False)['Close']
        returns = crypto_data.pct_change().dropna()
        
        mean_return = returns.mean()
        
        # 计算下行风险指标
        downside_returns = returns[returns < mean_return]
        if len(downside_returns) > 0:
            downside_deviation = np.sqrt(np.mean((downside_returns - mean_return)**2))
            downside_volatility = downside_deviation * np.sqrt(252)
            
            print(f"平均收益率: {mean_return:.4f}")
            print(f"下行标准差: {downside_deviation:.4f}")
            print(f"年化下行波动率: {float(downside_volatility):.2%}")
            print(f"下行风险天数: {len(downside_returns)}/{len(returns)}")
        
        # 计算不同目标收益率的半方差
        targets = [0, 0.01, 0.02]  # 0%, 1%, 2%日收益率
        print("\n不同目标收益率的半方差分析:")
        for target in targets:
            semivar, semi_std = calculate_target_semivariance(returns, target)
            print(f"目标收益率 {target:.1%}: 半标准差 {semi_std:.4f}")
        
        return returns
        
    except Exception as e:
        print(f"获取{symbol}数据时出错: {e}")
        return None


def calculate_target_semivariance(data, target):
    """计算目标半方差"""
    downside_data = data[data < target]
    if len(downside_data) > 0:
        semivariance = np.mean((downside_data - target)**2)
        semi_std = np.sqrt(semivariance)
        return semivariance, semi_std
    return 0, 0


def comprehensive_risk_analysis(symbol='SPY'):
    """综合风险指标分析"""
    print("=" * 60)
    print(f"5. 综合风险指标分析 - {symbol}")
    print("=" * 60)
    
    try:
        # 获取长期数据
        data = yf.download(symbol, period='5y', progress=False)['Close']
        returns = data.pct_change().dropna()
        
        # 计算各种风险指标
        risk_metrics = {}
        
        # 基础统计量
        risk_metrics['年化收益率'] = float(returns.mean() * 252)
        risk_metrics['年化波动率'] = float(returns.std() * np.sqrt(252))
        risk_metrics['夏普比率'] = risk_metrics['年化收益率'] / risk_metrics['年化波动率']
        
        # 极端风险
        risk_metrics['最大单日跌幅'] = float(returns.min())
        risk_metrics['VaR_5%'] = float(np.percentile(returns, 5))
        risk_metrics['CVaR_5%'] = float(returns[returns <= risk_metrics['VaR_5%']].mean())
        
        # 下行风险
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            risk_metrics['下行标准差'] = float(np.sqrt(np.mean(downside_returns**2)) * np.sqrt(252))
            risk_metrics['索蒂诺比率'] = risk_metrics['年化收益率'] / risk_metrics['下行标准差']
        
        # 最大回撤
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdowns = (cumulative - rolling_max) / rolling_max
        risk_metrics['最大回撤'] = float(drawdowns.min())
        
        # 输出结果
        print(f"{symbol} 风险指标汇总:")
        print("-" * 40)
        for metric, value in risk_metrics.items():
            if 'ratio' in metric.lower() or '比率' in metric:
                print(f"{metric}: {value:.3f}")
            elif '收益率' in metric or '波动率' in metric or '回撤' in metric or 'VaR' in metric or 'CVaR' in metric:
                print(f"{metric}: {value:.2%}")
            else:
                print(f"{metric}: {value:.4f}")
        
        # 可视化风险分析
        plt.figure(figsize=(15, 10))
        
        # 价格走势和回撤
        plt.subplot(2, 3, 1)
        plt.plot(data.index, data.values)
        plt.title(f'{symbol} 价格走势')
        plt.ylabel('价格')
        
        plt.subplot(2, 3, 2)
        plt.plot(drawdowns.index, drawdowns.values)
        plt.fill_between(drawdowns.index, drawdowns.values, 0, alpha=0.3, color='red')
        plt.title('回撤分析')
        plt.ylabel('回撤幅度')
        
        # 收益率分布
        plt.subplot(2, 3, 3)
        plt.hist(returns, bins=100, alpha=0.7, density=True)
        plt.axvline(risk_metrics['VaR_5%'], color='red', linestyle='--', label='5% VaR')
        plt.title('收益率分布')
        plt.xlabel('日收益率')
        plt.legend()
        
        # 滚动波动率
        plt.subplot(2, 3, 4)
        rolling_vol = returns.rolling(window=252).std() * np.sqrt(252)
        plt.plot(rolling_vol.index, rolling_vol.values)
        plt.title('滚动年化波动率')
        plt.ylabel('波动率')
        
        # 风险指标对比
        plt.subplot(2, 3, 5)
        risk_names = ['年化波动率', '下行标准差']
        risk_values = [risk_metrics.get(name, 0) for name in risk_names]
        plt.bar(risk_names, risk_values)
        plt.title('风险指标对比')
        plt.ylabel('年化波动率')
        plt.xticks(rotation=45)
        
        # 收益率箱线图（按年份）
        plt.subplot(2, 3, 6)
        yearly_returns = returns.groupby(returns.index.year)
        years = list(yearly_returns.groups.keys())
        yearly_data = [yearly_returns.get_group(year).values for year in years]
        plt.boxplot(yearly_data, labels=years)
        plt.title('年度收益率分布')
        plt.ylabel('日收益率')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        return risk_metrics
        
    except Exception as e:
        print(f"分析{symbol}时出错: {e}")
        return {}


def plot_risk_comparison():
    """比较不同资产的风险特征"""
    print("=" * 60)
    print("6. 多资产风险特征对比")
    print("=" * 60)
    
    symbols = ['SPY', 'QQQ', 'IWM', 'VNQ']  # 大盘、科技、小盘、房地产
    names = ['标普500', '纳斯达克100', '罗素2000', '房地产']
    
    plt.figure(figsize=(15, 10))
    colors = ['blue', 'green', 'red', 'orange']
    
    risk_comparison = {}
    
    for i, (symbol, name) in enumerate(zip(symbols, names)):
        try:
            data = yf.download(symbol, period='3y', progress=False)['Close']
            returns = data.pct_change().dropna()
            
            # 计算风险指标
            annual_return = float(returns.mean() * 252)
            annual_vol = float(returns.std() * np.sqrt(252))
            sharpe_ratio = annual_return / annual_vol if not np.isnan(annual_vol) and annual_vol > 0 else 0
            max_drawdown = float(((data / data.expanding().max()) - 1).min())
            
            risk_comparison[name] = {
                '年化收益率': annual_return,
                '年化波动率': annual_vol,
                '夏普比率': sharpe_ratio,
                '最大回撤': max_drawdown
            }
            
            # 绘制累计收益
            plt.subplot(2, 2, 1)
            cumulative_returns = (1 + returns).cumprod()
            plt.plot(cumulative_returns.index, cumulative_returns.values, 
                    label=name, color=colors[i])
            
            # 绘制滚动波动率
            plt.subplot(2, 2, 2)
            rolling_vol = returns.rolling(60).std() * np.sqrt(252)
            plt.plot(rolling_vol.index, rolling_vol.values, 
                    label=name, color=colors[i])
            
            # 绘制收益率分布
            plt.subplot(2, 2, 3)
            plt.hist(returns, bins=50, alpha=0.5, label=name, 
                    color=colors[i], density=True)
            
        except Exception as e:
            print(f"获取{symbol}数据失败: {e}")
    
    # 设置子图标题和标签
    plt.subplot(2, 2, 1)
    plt.title('累计收益对比')
    plt.ylabel('累计收益倍数')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.title('滚动波动率对比')
    plt.ylabel('年化波动率')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.title('收益率分布对比')
    plt.xlabel('日收益率')
    plt.ylabel('密度')
    plt.legend()
    
    # 风险指标对比表
    plt.subplot(2, 2, 4)
    plt.axis('off')
    
    # 创建风险指标对比表格
    if risk_comparison:
        table_data = []
        for asset_name, metrics in risk_comparison.items():
            row = [
                asset_name,
                f"{metrics['年化收益率']:.1%}",
                f"{metrics['年化波动率']:.1%}",
                f"{metrics['夏普比率']:.2f}",
                f"{metrics['最大回撤']:.1%}"
            ]
            table_data.append(row)
        
        headers = ['资产', '年化收益', '年化波动', '夏普比率', '最大回撤']
        table = plt.table(cellText=table_data, colLabels=headers,
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        plt.title('风险指标对比')
    
    plt.tight_layout()
    plt.show()
    
    return risk_comparison


def main():
    """主函数"""
    print("第7讲：方差 (Variance)")
    print("Quantopian量化分析56讲")
    print()
    
    # 1. 极差计算示例
    X = demonstrate_range()
    
    # 分析股票极差
    aapl_data = analyze_stock_range('AAPL')
    print()
    
    # 2. 平均绝对偏差
    calculate_mad(X)
    
    # 3. 方差和标准差
    analyze_variance_std(X)
    
    # 4. 股票收益率波动分析
    tsla_returns = analyze_stock_volatility('TSLA')
    if tsla_returns is not None:
        # 计算半方差
        calculate_semivariance(tsla_returns.values)
        print()
    
    # 5. 加密货币下行风险分析
    btc_returns = analyze_downside_risk('BTC-USD')
    print()
    
    # 6. 综合风险分析
    spy_metrics = comprehensive_risk_analysis('SPY')
    print()
    
    # 7. 多资产风险对比
    risk_comparison = plot_risk_comparison()
    
    # 总结要点
    print("\n" + "=" * 60)
    print("关键要点总结:")
    print("=" * 60)
    print("1. 标准差是衡量总体波动性的黄金标准")
    print("2. 半方差更关注下行风险，适合保守型投资者")
    print("3. 极差对异常值敏感，需结合其他指标分析")
    print("4. 金融数据分析需考虑年化处理（√252个交易日）")
    print("5. 不同资产类别需选择合适风险指标组合")
    print("6. VaR和CVaR可以量化极端风险")
    print("7. 回撤分析有助于理解投资期间的风险体验")
    print("=" * 60)


if __name__ == "__main__":
    main()