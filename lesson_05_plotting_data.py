"""
第5讲：数据可视化 (Plotting Data)
Quantopian量化分析56讲 - 第5节课

本课程学习如何使用Python进行数据可视化，包括：
- 直方图（Histogram）
- 累积直方图（Cumulative Histogram）
- 散点图（Scatter Plot）
- 折线图（Line Graph）

重要提醒：
- 图表适合用来提出假设，不适合直接验证假设
- 过去的趋势不代表未来
- 要避免确认偏误
"""

# 导入必要的库
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

# 设置中文字体（可选，如果需要显示中文标签）
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

def get_stock_data():
    """
    获取股票数据
    """
    print("正在获取股票数据...")
    
    # 设置时间范围
    start = '2014-01-01'
    end = '2015-01-01'
    
    # 使用 yfinance 获取 AAPL 和 MSFT 的数据
    raw_data = yf.download(['AAPL', 'MSFT'], start=start, end=end)
    print("原始数据结构：")
    print(raw_data.columns)
    
    # 检查数据结构并提取收盘价
    if isinstance(raw_data.columns, pd.MultiIndex):
        # 使用Close价格（yfinance默认auto_adjust=True，所以Close就是调整后价格）
        data = raw_data['Close']
    else:
        # 如果只有一只股票，结构会不同
        data = raw_data[['Close']]
    
    # 查看前几行数据
    print("数据预览：")
    print(data.head())
    print(f"数据形状：{data.shape}")
    
    return data

def plot_price_histogram(data):
    """
    图表 1：绘制股价的直方图
    """
    plt.figure(figsize=(10, 6))
    
    # 绘制 MSFT 股价的直方图，分成 20 个区间
    plt.hist(data['MSFT'], bins=20, alpha=0.7, color='blue')
    plt.xlabel('价格 ($)')
    plt.ylabel('观察到的天数')
    plt.title('2014年 MSFT 股价的频率分布')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print("直方图显示了MSFT股价在2014年的分布情况")

def plot_returns_histogram(data):
    """
    图表 2：绘制回报率的直方图
    """
    # 计算 MSFT 的每日回报率（移除第一行，因为第一天没有前一天数据）
    R = data['MSFT'].pct_change()[1:]
    
    plt.figure(figsize=(10, 6))
    plt.hist(R, bins=20, alpha=0.7, color='green')
    plt.xlabel('回报率')
    plt.ylabel('观察到的天数')
    plt.title('2014年 MSFT 回报的频率分布')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print(f"回报率统计：")
    print(f"平均回报率: {R.mean():.4f}")
    print(f"标准差: {R.std():.4f}")
    print(f"最大回报率: {R.max():.4f}")
    print(f"最小回报率: {R.min():.4f}")

def plot_cumulative_histogram(data):
    """
    图表 3：绘制累积直方图
    """
    # 计算 MSFT 的每日回报率
    R = data['MSFT'].pct_change()[1:]
    
    plt.figure(figsize=(10, 6))
    plt.hist(R, bins=20, cumulative=True, alpha=0.7, color='orange')
    plt.xlabel('回报率')
    plt.ylabel('累积观察天数')
    plt.title('2014年 MSFT 回报的累积分布')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print("累积直方图显示了回报率的累积分布")

def plot_scatter_plots(data):
    """
    图表 4：绘制散点图
    """
    # 散点图1：股价散点图
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(data['MSFT'], data['AAPL'], alpha=0.6)
    plt.xlabel('MSFT 价格 ($)')
    plt.ylabel('AAPL 价格 ($)')
    plt.title('2014年 MSFT vs AAPL 每日股价')
    plt.grid(True, alpha=0.3)
    
    # 散点图2：回报散点图
    R_msft = data['MSFT'].pct_change()[1:]
    R_aapl = data['AAPL'].pct_change()[1:]
    
    plt.subplot(1, 2, 2)
    plt.scatter(R_msft, R_aapl, alpha=0.6, color='red')
    plt.xlabel('MSFT 回报率')
    plt.ylabel('AAPL 回报率')
    plt.title('2014年 MSFT vs AAPL 每日回报')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 计算相关性
    correlation = np.corrcoef(R_msft, R_aapl)[0, 1]
    print(f"MSFT和AAPL回报率的相关系数: {correlation:.4f}")

def plot_line_graphs(data):
    """
    图表 5：绘制折线图
    """
    # 折线图1：股价时间序列
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(data['MSFT'], label='MSFT', linewidth=2)
    plt.plot(data['AAPL'], label='AAPL', linewidth=2)
    plt.ylabel('价格 ($)')
    plt.title('2014年股价走势')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 折线图2：MSFT回报时间序列
    R_msft = data['MSFT'].pct_change()[1:]
    plt.subplot(2, 2, 2)
    plt.plot(R_msft, color='blue', linewidth=1)
    plt.ylabel('回报率')
    plt.title('MSFT 每日回报')
    plt.grid(True, alpha=0.3)
    
    # 折线图3：AAPL回报时间序列
    R_aapl = data['AAPL'].pct_change()[1:]
    plt.subplot(2, 2, 3)
    plt.plot(R_aapl, color='red', linewidth=1)
    plt.ylabel('回报率')
    plt.title('AAPL 每日回报')
    plt.grid(True, alpha=0.3)
    
    # 折线图4：标准化价格比较
    plt.subplot(2, 2, 4)
    # 标准化到起始价格为1
    normalized_msft = data['MSFT'] / data['MSFT'].iloc[0]
    normalized_aapl = data['AAPL'] / data['AAPL'].iloc[0]
    
    plt.plot(normalized_msft, label='MSFT (标准化)', linewidth=2)
    plt.plot(normalized_aapl, label='AAPL (标准化)', linewidth=2)
    plt.ylabel('标准化价格')
    plt.title('标准化股价比较')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def comprehensive_analysis(data):
    """
    综合分析示例
    """
    print("\n=== 综合数据分析 ===")
    
    # 计算基本统计信息
    print("\n股价基本统计：")
    print(data.describe())
    
    # 计算回报率
    returns = data.pct_change()[1:]
    print("\n回报率基本统计：")
    print(returns.describe())
    
    # 计算年化收益率和波动率
    trading_days = 252
    annual_return_msft = returns['MSFT'].mean() * trading_days
    annual_return_aapl = returns['AAPL'].mean() * trading_days
    annual_vol_msft = returns['MSFT'].std() * np.sqrt(trading_days)
    annual_vol_aapl = returns['AAPL'].std() * np.sqrt(trading_days)
    
    print(f"\n年化指标：")
    print(f"MSFT - 年化收益率: {annual_return_msft:.2%}, 年化波动率: {annual_vol_msft:.2%}")
    print(f"AAPL - 年化收益率: {annual_return_aapl:.2%}, 年化波动率: {annual_vol_aapl:.2%}")

def main():
    """
    主函数：运行所有图表示例
    """
    print("=" * 50)
    print("第5讲：数据可视化 (Plotting Data)")
    print("Quantopian量化分析56讲")
    print("=" * 50)
    
    try:
        # 1. 获取数据
        data = get_stock_data()
        
        # 2. 绘制各种图表
        print("\n1. 绘制股价直方图...")
        plot_price_histogram(data)
        
        print("\n2. 绘制回报率直方图...")
        plot_returns_histogram(data)
        
        print("\n3. 绘制累积直方图...")
        plot_cumulative_histogram(data)
        
        print("\n4. 绘制散点图...")
        plot_scatter_plots(data)
        
        print("\n5. 绘制折线图...")
        plot_line_graphs(data)
        
        # 3. 综合分析
        comprehensive_analysis(data)
        
        print("\n" + "=" * 50)
        print("课程完成！")
        print("重要提醒：")
        print("- 图表帮助我们探索数据和提出假设")
        print("- 不要用图表直接验证假设")
        print("- 过去的表现不代表未来结果")
        print("=" * 50)
        
    except Exception as e:
        print(f"发生错误: {e}")
        print("请检查网络连接和yfinance库是否正确安装")

if __name__ == "__main__":
    main()