print("RUNNING ON CPU")
from library import config, utils, broker_funcs, portfolio
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
#from fbm.fbm import fbm
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

for t in range(config.tinit+1,config.tinit+1+config.simsteps):
    broker, totalOrders = broker_funcs.thresholdBrokerage(traderIDs, t, broker, totalOrders)
    broker, transactions = broker_funcs.instantMatch(traderIDs, broker, transactions)
    portfolio.priceChange(time=t)
    print("New threshold " +config.nportfs,config.threshold+" | ", t)

t1 = time.localtime()
t1str = time.strftime("%H:%M:%S",t1)



with open('./results/traderIDs_'+config.nportfs+'_1000_'+config.threshold + '.pkl', 'wb') as f:
    pickle.dump(traderIDs, f, pickle.HIGHEST_PROTOCOL)

print("CPU RUN TIME | nportfs: ", config.nportfs)
print(t0str)
print(t1str)
print(config.config)
TstockPool, ThurstPool = portfolio.stockChars()

transactions.to_csv('./results/transactions_'+config.nportfs+'_'+config.simsteps+'_'+config.threshold+'newthreshold.csv')
totalOrders.to_csv('./results/totalOrders_'+config.nportfs+'_'+config.simsteps+'_'+config.threshold+'newthreshold.csv')
np.save('./results/stockPool_'+config.nportfs+'_'+config.simsteps+'_'+config.threshold+'newthreshold.npy',TstockPool)
np.save('./results/hurstPool_'+config.nportfs+'_'+config.simsteps+'_'+config.threshold+'newthreshold.npy',ThurstPool)
conf = open('./results/config_'+config.nportfs+'_'+config.simsteps+'_'+config.threshold+'newthresholded' + '.txt',"w")
conf.write(str(config.config))
conf.close()
