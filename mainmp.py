print("RUNNING ON CPU MP")
from libraryMP import config, utils, broker_funcs, portfolio
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

for t in range(993,1003):
    broker, totalOrders = broker_funcs.brokerage(traderIDs, t, broker, totalOrders)
    broker, transactions = broker_funcs.instantMatch(traderIDs, broker, transactions)
    portfolio.priceChange(time=t)
    print(t)


Ttransactions = pd.DataFrame()
TtotalOrders = pd.DataFrame()
Tbroker = pd.DataFrame()

for key,portf in traderIDs.items():
    portf.reset(ptile=70)

for t in range (993,4592):
    Tbroker, TtotalOrders = broker_funcs.thresholdBrokerage(traderIDs, t, Tbroker, TtotalOrders)
    Tbroker, Ttransactions = broker_funcs.instantMatch(traderIDs, Tbroker, Ttransactions)
    portfolio.priceChange(time=t)
    print(t)


t1 = time.localtime()
t1str = time.strftime("%H:%M:%S",t1)



# with open('./results/traderIDs_1000_threshold' + '.pkl', 'wb') as f:
#     pickle.dump(traderIDs, f, pickle.HIGHEST_PROTOCOL)

print("CPU MP RUN TIME | nportfs: ", config.nportfs)
print(t0str)
print(t1str)

# TstockPool, ThurstPool = portfolio.stockChars()

# Ttransactions.to_csv('./results/Ttransactions_1000_threshold.csv')
# TtotalOrders.to_csv('./results/TtotalOrders_1000_threshold.csv')
# np.save('./results/TstockPool_1000.npy',TstockPool)
# np.save('./results/ThurstPool_1000.npy',ThurstPool)
# conf = open('./results/config_1000_pricechange_thresholded' + '.txt',"w")
# conf.write(str(config.config))
# conf.close()