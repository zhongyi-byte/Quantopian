from tokenize import endpats
from turtledemo.penrose import start

import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

np.random.seed(42)

X = np.sort(np.random.randint(100, size=100))
print("数据集：\n", X)
print("Range：", np.ptp(X))


# 获取特斯拉2022年收益率
tsla = yf.download('TSLA', start='2022-01-01', end='2023-01-01')['Close']
returns = tsla.pct_change().dropna()

plt.figure(figsize=(10,5))
plt.plot(returns.cumsum(), label='累计收益')
plt.fill_between(returns.index, returns, alpha=0.3, color='gray', label='日波动')
plt.title("特斯拉2022年收益率波动")
plt.legend()
plt.show()

print(f"\n特斯拉收益率标准差: {returns.std():.4f}")
print(f"年化波动率: {returns.std()*np.sqrt(252):.2%}")

