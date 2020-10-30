import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from library import utils, config

# cacasde algorithm I showed joern back in aug 25


# 1. start with 1 portfolio's set of stocks 
# 2. look at all portoflios that also hold those stocks
# 3. add their cascades to the cascade and repeat
# 4. new seed portfolio would be trades of stocks that are not within the other cascades stock pool
def findCascades(TtotalOrders, maxtimeeff):

    cascades = {}
    numCascade = 0
    cascadeStocks = {}
    cascadePortfs = {}
    cascadeTime = {}
    cascadeTf = {}
    cascades = {}
    # maxtimeeff = 2
    while len(TtotalOrders) > 0:
    #     totalTimeOrders = TtotalOrders[TtotalOrders['time']==t]
        # seed with first portfolio
        print("OG Orders left: ", len(TtotalOrders))
        seed = (TtotalOrders.iloc[0]['portfolio'])
        t0 = TtotalOrders.iloc[0]['time']
        seedCascade = findPortfOrderCascades(TtotalOrders, seed, t0, maxSep = 1)
        cascadeStocks[numCascade] = np.asarray(seedCascade['stock'])
        cascadePortfs[numCascade] = np.asarray(seed)
        cascadeTime[numCascade] = t0
        cascades[numCascade] = seedCascade
        stockTimes = {}
        for row in seedCascade.iterrows():
            stockTimes[row[1]['stock']] = row[1]['time'] + abs(row[1]['order'])*maxtimeeff
        assert len(seedCascade) > 0
        TtotalOrders = TtotalOrders[~TtotalOrders.isin(seedCascade)].dropna()
        # NEW VERSION (PORTFOLIO PERSPECTIVE)
        for childPortf in list(TtotalOrders['portfolio'].unique()):
            match = False
            if any(np.isin(cascadeStocks[numCascade],traderIDs[childPortf].stocks)):
    #             print("match: ", childPortf)
                if childPortf not in cascadePortfs[numCascade]:
                    childPortfCascade = findPortfOrderCascades(TtotalOrders,childPortf,t0, maxSep = 1)
                    for row in childPortfCascade.iterrows():
                        if row[1]['stock'] in stockTimes:
                            if row[1]['time'] <= stockTimes[row[1]['stock']]:
                                match = True
                                break 
                    if match == True:        
                        cascadePortfs[numCascade] = np.append(cascadePortfs[numCascade], childPortf)
                        cascades[numCascade] = pd.concat([cascades[numCascade],childPortfCascade])
                        TtotalOrders = TtotalOrders[~TtotalOrders.isin(childPortfCascade)].dropna()
    #                     for childStock in list(childPortfCascade['stock'].unique()):
    #                         if childStock not in cascadeStocks[numCascade]:
    #                             cascadeStocks[numCascade] = np.append(cascadeStocks[numCascade], childStock)
                        for row in childPortfCascade.iterrows():
                            if row[1]['stock'] in stockTimes:
                                stockTimes[row[1]['stock']] += abs(row[1]['order'])*maxtimeeff
                            else:
                                stockTimes[row[1]['stock']] = row[1]['time'] + abs(row[1]['order'])*maxtimeeff
                                cascadeStocks[numCascade] = np.append(cascadeStocks[numCascade], row[1]['stock'])
        print("Cascade length: ",len(cascades[numCascade]))
        print("New Orders left: ", len(TtotalOrders))
        cascadeTf[numCascade] = cascades[numCascade]['time'].max()
        print("making new cascade")
        numCascade += 1
        


        
    print(cascadeStocks)
    print(cascadePortfs)
    print(cascadeTime)


def findStockOrderCascades(stockOrders, cascade, maxSep = 2):
    """
    finds all subsequent activity of a single stock (time threshold)
    inpurt filtered stockOrders table with time > _time
    works on stockOrder tables, returns value of stock moved
    legacy Aug1
    """
    if len(stockOrders) > 0 :
        ToS = stockOrders['time'].unique()
        sep = np.diff(ToS)
        #print(sep)
        #print(max(sep))
        keySep = np.where(sep>maxSep)[0]
        #print(keySep)
        if len(keySep)>0:
            endtime = ToS[keySep[0]] #end of this cascade
        else:
            endtime = ToS[-1]
        cascade = pd.concat([cascade,stockOrders[stockOrders['time']<=endtime]])
        return cascade
    else:
        print() 
    
def findPortfOrderCascades(TtotalOrders, portf, t0, maxSep = 1):
    """
    finds all subsequent activity traded by a portfolio (time threshold)
    input TtotalOrders, the portfolio, and minimum time
    FOR REAL CASCADES Aug18
    """

    portfOrders = TtotalOrders[TtotalOrders['portfolio']==portf]
    portfOrders = portfOrders[portfOrders['time']>=t0]
    if len(portfOrders)>0:
        ToS = portfOrders['time'].unique()
        sep = np.diff(ToS)
        #print(sep)
        #print(max(sep))
        keySep = np.where(sep>maxSep)[0]
        #print(keySep)
        if len(keySep)>0:
            endtime = ToS[keySep[0]] #end of this cascade
        else:
            endtime = ToS[-1]
        portfCascade = portfOrders[portfOrders['time']<=endtime]
        return portfCascade
    else:
        return pd.DataFrame()