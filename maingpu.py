print("RUNNING ON GPU")
from libraryGPU import config, utils, broker_funcs, portfolio
import cupy as np
import cudf as pd
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

for t in range(993,1005):
    # 50 traders
    broker, totalOrders = broker_funcs.brokerage(traderIDs, t, broker, totalOrders)
    broker, transactions = broker_funcs.instantMatch(traderIDs, broker, transactions)
    print(t)

t1 = time.localtime()
t1str = time.strftime("%H:%M:%S",t1)



with open('./results/traderIDs_50_gputest_nothreshold' + '.pkl', 'wb') as f:
    pickle.dump(traderIDs, f, pickle.HIGHEST_PROTOCOL)

print("GPU RUN TIME")
print(t0str)
print(t1str)

stockPool, hurstPool = portfolio.stockChars()

transactions.to_csv('./results/transactions_50_gputest_nothreshold.csv')
totalOrders.to_csv('./results/totalOrders_50_gputest_nothreshold.csv')
np.save('./results/stockPool_gpu.npy',stockPool)
np.save('./results/hurstPool_gpu.npy',hurstPool)
