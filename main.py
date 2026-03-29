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
