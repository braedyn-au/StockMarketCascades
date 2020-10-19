print("RUNNING ON CPU")
from library import config, utils, broker_funcs, portfolio
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
#from fbm.fbm import fbm
import time
import pickle
import os 

#assert config.changePrice == True

print(config.config)
rpath = 'results/nopricechange/'+str(config.overlapMin)+'-'+str(config.overlapMax)+'/'
if not os.path.exists(rpath):
        os.makedirs(rpath)
print('saving to: ', rpath)
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
    print("New threshold " ,config.nportfs,config.threshold," | ", t)

t1 = time.localtime()
t1str = time.strftime("%H:%M:%S",t1)


with open(rpath+'traderIDs_'+str(config.nportfs)+'_'+str(config.simsteps)+'_'+str(config.threshold) + '_new2threshold_fixedhurst.pkl', 'wb') as f:
    pickle.dump(traderIDs, f, pickle.HIGHEST_PROTOCOL)

print("CPU RUN TIME | nportfs: ", config.nportfs)
print(t0str)
print(t1str)
print(config.config)
TstockPool, ThurstPool = portfolio.stockChars()

transactions.to_csv(rpath+'transactions_'+str(config.nportfs)+'_'+str(config.simsteps)+'_'+str(config.threshold)+'_new2threshold_fixedhurst.csv')
totalOrders.to_csv(rpath+'totalOrders_'+str(config.nportfs)+'_'+str(config.simsteps)+'_'+str(config.threshold)+'_new2threshold_fixedhurst.csv')
np.save(rpath +'stockPool_'+str(config.nportfs)+'_'+str(config.simsteps)+'_'+str(config.threshold)+'_new2threshold_fixedhurst.npy',TstockPool)
np.save(rpath +'hurstPool_'+str(config.nportfs)+'_'+str(config.simsteps)+'_'+str(config.threshold)+'_new2threshold_fixedhurst.npy',ThurstPool)
conf = open(rpath+'config_'+str(config.nportfs)+'_'+str(config.simsteps)+'_'+str(config.threshold)+'_new2thresholded_fixedhurst' + '.txt',"w")
conf.write(str(config.config))
conf.close()
