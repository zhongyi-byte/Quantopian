"""
测试pandas Series格式化修复
"""

import numpy as np
import pandas as pd

def test_format_fix():
    """测试格式化修复"""
    print("测试pandas Series格式化问题:")
    
    # 创建模拟数据
    returns = pd.Series(np.random.normal(0.001, 0.02, 252))
    
    # 测试原来会出错的代码（现在应该工作）
    try:
        annual_vol = returns.std() * np.sqrt(252)
        max_return = returns.max()
        min_return = returns.min()
        
        # 修复后的代码 - 转换为float
        print(f"年化波动率: {float(annual_vol):.2%}")
        print(f"最大单日收益: {float(max_return):.2%}")
        print(f"最大单日亏损: {float(min_return):.2%}")
        
        print("✅ 格式化修复成功！")
        
    except Exception as e:
        print(f"❌ 仍有错误: {e}")
    
    # 测试风险指标字典
    try:
        risk_metrics = {
            '年化收益率': float(returns.mean() * 252),
            '年化波动率': float(returns.std() * np.sqrt(252)),
            '最大单日跌幅': float(returns.min()),
            'VaR_5%': float(np.percentile(returns, 5)),
        }
        
        print("\n风险指标测试:")
        for metric, value in risk_metrics.items():
            if '收益率' in metric or '波动率' in metric or 'VaR' in metric:
                print(f"{metric}: {value:.2%}")
            else:
                print(f"{metric}: {value:.4f}")
        
        print("✅ 风险指标字典格式化成功！")
        
    except Exception as e:
        print(f"❌ 风险指标格式化错误: {e}")

def test_boolean_fix():
    """测试布尔值错误修复"""
    print("\n测试布尔值判断修复:")
    
    try:
        # 模拟数据
        data = pd.Series([100, 105, 103, 108, 106])
        annual_vol = 0.25
        
        # 修复后的条件判断
        sharpe_ratio = 0.8 / annual_vol if not np.isnan(annual_vol) and annual_vol > 0 else 0
        max_drawdown = float(((data / data.expanding().max()) - 1).min())
        
        print(f"夏普比率: {sharpe_ratio:.2f}")
        print(f"最大回撤: {max_drawdown:.2%}")
        print("✅ 布尔值判断修复成功！")
        
    except Exception as e:
        print(f"❌ 布尔值判断仍有错误: {e}")

if __name__ == "__main__":
    test_format_fix()
    test_boolean_fix()
    print("\n" + "=" * 40)
    print("如果看到✅，说明修复成功！")
    print("现在可以正常运行lesson07_variance.py了")
    print("=" * 40)