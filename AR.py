import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima_model import AR
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.api import qqplot

from mpi4py import MPI


dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
SPYdata = pd.read_csv('SPY.csv', parse_dates=['Date'], index_col=['Date'],date_parser=dateparse)
IWVdata = pd.read_csv('IWV.csv', parse_dates=['Date'], index_col=['Date'],date_parser=dateparse)


# calculate the percentage return of SPY and IWV
SPYreturn = SPYdata['Adj Close'].diff() / SPYdata['Adj Close']
IWVreturn = IWVdata['Adj Close'].diff() / IWVdata['Adj Close']


# get X_t = IWYreturn - SPYreturn
targetData = IWVreturn - SPYreturn
targetData = targetData[~targetData.isnull()]

# draw the data
f1 = plt.figure(facecolor='white')
f1.add_subplot(211)
plt.plot(SPYreturn, color="blue", linewidth=2.5, linestyle="-", label="SPY return")
plt.plot(IWVreturn, color="red",  linewidth=2.5, linestyle="-", label="IWV return")
plt.xlabel("Time")
plt.ylabel("Return")
plt.legend(loc='upper left')

f1.add_subplot(212)
plt.plot(targetData, color="blue",  linewidth=2.5, linestyle="-", label="Difference of Return")
plt.xlabel("Time")
plt.ylabel("Difference of Return")


# stationory
stationoryResult = adfuller(targetData)
# print 'adf: ', stationoryResult[0]
# print 'p-value: ', stationoryResult[1]
# print 'Critical values: ', stationoryResult[4]
# if stationoryResult[0]> stationoryResult[4]['5%']:
#     print 'Time Series is nonstationary'
# else:
#     print 'Time Series is stationary'

# acf, pacf
f2 = plt.figure(facecolor='white')
ax1 = f2.add_subplot(211)
plot_acf(targetData, lags=40, ax=ax1)
ax2 = f2.add_subplot(212)
plot_pacf(targetData, lags=40, ax=ax2)
# plt.show()

# according to the problem, only use AR(p) p belongs to (30,35)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

AR_mod = lambda x: ARMA(targetData,(x,0)).fit(disp = 0, method = 'mle')
aic_list = AR_mod(rank).aic
print "%i|%i|%i" % (rank, aic_list, rank)


'''
# confirm the lag
p_index = aic_list.index(min(aic_list))
AR = AR_mod(p_index)


# residual analysis
resid = AR.resid
f3 = plt.figure(facecolor='white')
ax1 = f3.add_subplot(211)
plot_acf(resid.values.squeeze(), lags=40, ax=ax1)
ax2 = f3.add_subplot(212)
plot_pacf(resid, lags=40, ax=ax2)
# plt.show()

# qq plot
f4 = plt.figure(facecolor='white')
ax = f4.add_subplot(111)
qqplot(resid, line='q', ax=ax, fit=True)
plt.show()

# L-B test, which shows the residual is white noise
print acorr_ljungbox(resid,lags = 20)[1]
'''



