"""
测试修复后的方差计算代码 - 简化版
"""

import numpy as np
import yfinance as yf
import warnings

warnings.filterwarnings('ignore')

def test_stock_analysis():
    """测试股票分析功能"""
    print("测试TSLA收益率分析:")
    
    try:
        # 获取TSLA数据
        tsla_data = yf.download('TSLA', start='2022-01-01', end='2023-01-01', progress=False)['Close']
        returns = tsla_data.pct_change().dropna()
        
        # 计算波动率指标（修复后）
        daily_vol = returns.std()
        annual_vol = daily_vol * np.sqrt(252)
        
        print(f"日收益率标准差: {daily_vol:.4f}")
        print(f"年化波动率: {float(annual_vol):.2%}")  # 修复：转换为float
        print(f"最大单日收益: {float(returns.max()):.2%}")  # 修复：转换为float
        print(f"最大单日亏损: {float(returns.min()):.2%}")  # 修复：转换为float
        print("✅ TSLA分析成功")
        
    except Exception as e:
        print(f"❌ TSLA分析失败: {e}")
    
    print()

def test_btc_analysis():
    """测试BTC分析功能"""
    print("测试BTC-USD下行风险分析:")
    
    try:
        # 获取BTC数据
        btc_data = yf.download('BTC-USD', start='2021-01-01', end='2022-01-01', progress=False)['Close']
        returns = btc_data.pct_change().dropna()
        
        mean_return = returns.mean()
        downside_returns = returns[returns < mean_return]
        
        if len(downside_returns) > 0:
            downside_deviation = np.sqrt(np.mean((downside_returns - mean_return)**2))
            downside_volatility = downside_deviation * np.sqrt(252)
            
            print(f"平均收益率: {mean_return:.4f}")
            print(f"下行标准差: {downside_deviation:.4f}")
            print(f"年化下行波动率: {float(downside_volatility):.2%}")  # 修复：转换为float
            print("✅ BTC分析成功")
        
    except Exception as e:
        print(f"❌ BTC分析失败: {e}")
    
    print()

def test_spy_comprehensive():
    """测试SPY综合分析"""
    print("测试SPY综合风险分析:")
    
    try:
        # 获取SPY数据
        data = yf.download('SPY', period='2y', progress=False)['Close']
        returns = data.pct_change().dropna()
        
        # 计算各种风险指标（修复后）
        risk_metrics = {}
        
        # 基础统计量 - 修复：转换为float
        risk_metrics['年化收益率'] = float(returns.mean() * 252)
        risk_metrics['年化波动率'] = float(returns.std() * np.sqrt(252))
        risk_metrics['夏普比率'] = risk_metrics['年化收益率'] / risk_metrics['年化波动率']
        
        # 极端风险 - 修复：转换为float
        risk_metrics['最大单日跌幅'] = float(returns.min())
        risk_metrics['VaR_5%'] = float(np.percentile(returns, 5))
        risk_metrics['CVaR_5%'] = float(returns[returns <= risk_metrics['VaR_5%']].mean())
        
        # 最大回撤 - 修复：转换为float
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdowns = (cumulative - rolling_max) / rolling_max
        risk_metrics['最大回撤'] = float(drawdowns.min())
        
        print("SPY 风险指标汇总:")
        print("-" * 40)
        for metric, value in risk_metrics.items():
            if 'ratio' in metric.lower() or '比率' in metric:
                print(f"{metric}: {value:.3f}")
            elif '收益率' in metric or '波动率' in metric or '回撤' in metric or 'VaR' in metric or 'CVaR' in metric:
                print(f"{metric}: {value:.2%}")
            else:
                print(f"{metric}: {value:.4f}")
        
        print("✅ SPY分析成功")
        
    except Exception as e:
        print(f"❌ SPY分析失败: {e}")
    
    print()

def test_multi_asset():
    """测试多资产对比"""
    print("测试多资产风险对比:")
    
    symbols = ['SPY', 'QQQ']  # 简化测试只用两个
    names = ['标普500', '纳斯达克100']
    
    for symbol, name in zip(symbols, names):
        try:
            data = yf.download(symbol, period='1y', progress=False)['Close']
            returns = data.pct_change().dropna()
            
            # 计算风险指标 - 修复：转换为float
            annual_return = float(returns.mean() * 252)
            annual_vol = float(returns.std() * np.sqrt(252))
            sharpe_ratio = annual_return / annual_vol if not np.isnan(annual_vol) and annual_vol > 0 else 0
            max_drawdown = float(((data / data.expanding().max()) - 1).min())
            
            print(f"{name} ({symbol}):")
            print(f"  年化收益率: {annual_return:.2%}")
            print(f"  年化波动率: {annual_vol:.2%}")
            print(f"  夏普比率: {sharpe_ratio:.2f}")
            print(f"  最大回撤: {max_drawdown:.2%}")
            print(f"✅ {name}分析成功")
            
        except Exception as e:
            print(f"❌ {name}分析失败: {e}")
    
    print()

def main():
    """主测试函数"""
    print("=" * 50)
    print("测试修复后的方差分析代码")
    print("=" * 50)
    
    test_stock_analysis()
    test_btc_analysis() 
    test_spy_comprehensive()
    test_multi_asset()
    
    print("=" * 50)
    print("测试完成！如果看到✅说明修复成功")
    print("=" * 50)

if __name__ == "__main__":
    main()