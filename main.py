print("RUNNING ON CPU")
from library import config, utils, broker_funcs, portfolio
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from fbm.fbm import fbm
import time
import pickle

assert config.changePrice == True

print(config.config)

t0 = time.localtime()
t0str = time.strftime("%H:%M:%S",t0)


traderIDs = portfolio.portfGen()

transactions = pd.DataFrame()
totalOrders = pd.DataFrame()
broker = pd.DataFrame()

for t in range(993,4592):
    broker, totalOrders = broker_funcs.thresholdBrokerage(traderIDs, t, broker, totalOrders)
    broker, transactions = broker_funcs.instantMatch(traderIDs, broker, transactions)
    portfolio.priceChange(time=t)
    print("New threshold 500", t)

# with open('./results/traderIDs_cpu_nothreshold' + '.pkl', 'wb') as f:
#     pickle.dump(traderIDs, f, pickle.HIGHEST_PROTOCOL)

# Ttransactions = pd.DataFrame()
# TtotalOrders = pd.DataFrame()
# Tbroker = pd.DataFrame()

# for key,portf in traderIDs.items():
#     portf.reset(ptile=70)

# for t in range (993,4592):
#     Tbroker, TtotalOrders = broker_funcs.thresholdBrokerage(traderIDs, t, Tbroker, TtotalOrders)
#     Tbroker, Ttransactions = broker_funcs.instantMatch(traderIDs, Tbroker, Ttransactions)
#     portfolio.priceChange(time=t)
#     print(t)


t1 = time.localtime()
t1str = time.strftime("%H:%M:%S",t1)



with open('./results/traderIDs_500_newthreshold' + '.pkl', 'wb') as f:
    pickle.dump(traderIDs, f, pickle.HIGHEST_PROTOCOL)

print("CPU RUN TIME | nportfs: ", config.nportfs)
print(t0str)
print(t1str)

TstockPool, ThurstPool = portfolio.stockChars()

transactions.to_csv('./results/transactions_500_newthreshold.csv')
totalOrders.to_csv('./results/totalOrders_500_newthreshold.csv')
np.save('./results/stockPool_500_newthreshold.npy',TstockPool)
np.save('./results/hurstPool_500_newthreshold.npy',ThurstPool)
conf = open('./results/config_500_newthresholded' + '.txt',"w")
conf.write(str(config.config))
conf.close()