from scipy.spatial.transform import rotation
import pandas as pd
import numpy as np 
from matplotlib import pyplot as plt 
import yfinance as yf 
import seaborn as sns

stocks = [
    'AKBNK.IS', 'ISCTR.IS', 'GARAN.IS', 'YKBNK.IS', 'HALKB.IS', 'VAKBN.IS',
    'KCHOL.IS', 'SAHOL.IS',
    'EREGL.IS', 'SISE.IS', 'TUPRS.IS',
    'FROTO.IS', 'TOASO.IS',
    'BIMAS.IS', 'MGROS.IS',
    'THYAO.IS', 'TCELL.IS',
    'ASELS.IS', 'ENKAI.IS'
]

startDate = '2024-03-01'
endDate = '2026-03-01'

data = yf.download(stocks, start=startDate, end=endDate)['Close']
data = data.dropna()

limit = data.shape[0]
daily_diff = np.zeros((limit, len(stocks)))
daily_differences = pd.DataFrame(daily_diff, columns=stocks)

for stock in stocks:
    for i in range(limit):
        daily_differences.loc[i, stock] = data[stock].iloc[i] - data[stock].iloc[i-1]
covariance_matrix = daily_differences.cov()
correlation_matrix = daily_differences.corr() 

top10_corr = correlation_matrix.unstack().sort_values(ascending=False).drop_duplicates()
top10_corr = top10_corr[top10_corr != 1]
top10_corr = top10_corr.sort_values(ascending=False).head(10).reset_index()
top10_corr.columns = ["Stock1", "Stock2", "Correlation"]
top10_corr["Pairs"] = top10_corr["Stock1"] + " - " + top10_corr["Stock2"]

fig, axes = plt.subplots(2, 1, figsize=(12, 8))
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap='coolwarm',
    fmt=".2f",
    linewidths=0.5,
    linecolor="white",
    annot_kws={"size": 8, "color": "black"},
    cbar_kws={"shrink": 0.8},
    ax=axes[0]
)
axes[0].set_title(
    'BIST Core Stocks: Correlation Heatmap',
    fontsize=14,
    weight="bold",
    pad=10
)
axes[0].tick_params(axis='x', rotation=45, labelsize=9, colors="#333333")
axes[0].tick_params(axis='y', rotation=0, labelsize=9, colors="#333333")

sns.barplot(
    data=top10_corr,
    x="Correlation",
    y="Pairs",
    palette="viridis",
    ax=axes[1]
)
axes[1].set_title(
    "Top 10 Correlated Stock Pairs",
    fontsize=13,
    weight="bold",
    pad=10
)
axes[1].set_xlabel("Correlation", fontsize=10, color="#333333")
axes[1].set_ylabel("")
axes[1].tick_params(axis='x', labelsize=9, colors="#333333")
axes[1].tick_params(axis='y', labelsize=9, colors="#333333")


axes[1].grid(axis="x", linestyle="--", alpha=0.3)
for i, v in enumerate(top10_corr["Correlation"]):
    axes[1].text(v + 0.005, i, f"{v:.2f}", va="center", fontsize=9, color="#222222")

plt.tight_layout()
plt.show()