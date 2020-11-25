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
def findCascades(TtotalOrders, traderIDs, maxtimeeff = 2, maxSep = 2):
    """
    find causal cascades
    """
    cascades = {}
    numCascade = 0
    cascadeStocks = {}
    cascadePortfs = {}
    cascadeTime = {}
    cascadeTf = {}
    while len(TtotalOrders) > 0:
        print("OG Orders left: ", len(TtotalOrders))
        seed = (TtotalOrders.iloc[0]['portfolio'])
        t0 = TtotalOrders.iloc[0]['time']
        seedCascade = findPortfOrderCascade(TtotalOrders, seed, t0, maxSep = maxSep)
        cascadeStocks[numCascade] = np.asarray(seedCascade['stock'])
        cascadePortfs[numCascade] = np.asarray(seed)
        cascadeTime[numCascade] = t0
        cascades[numCascade] = seedCascade
        # Dictionary keeping track of how long a stock is still "affected" from previous trade
        stockTimes = {}
        for row in seedCascade.iterrows():
            stockTimes[row[1]['stock']] = row[1]['time'] + abs(row[1]['order'])*maxtimeeff

        # check first cascade is greater than -
        assert len(seedCascade) > 0

        # remove rows included in first cascade from the order list
        TtotalOrders = TtotalOrders[~TtotalOrders.isin(seedCascade)].dropna()

        # NEW VERSION (PORTFOLIO PERSPECTIVE)
        for childPortf in list(TtotalOrders['portfolio'].unique()):
            match = False

            # see if the current portfolio has overlapping stocks with the ongoing cascade 
            if any(np.isin(cascadeStocks[numCascade],traderIDs[childPortf].stocks)):
                # check if overlapping portfolio is already included in the ongoing cascade
                if childPortf not in cascadePortfs[numCascade]:
                    # include new portfolio into the cascade along with all of its cascading trading activity
                    childPortfCascade = findPortfOrderCascade(TtotalOrders,childPortf,t0, maxSep = maxSep)
                    # check if trading activity falls into the time affect window of ongoing cascade
                    for row in childPortfCascade.iterrows():
                        if row[1]['stock'] in stockTimes:
                            if row[1]['time'] <= stockTimes[row[1]['stock']]:
                                match = True
                                break 
                    if match == True:        
                        # add new cascade to the ongoing cascade and remove rows from the orderList
                        cascadePortfs[numCascade] = np.append(cascadePortfs[numCascade], childPortf)
                        cascades[numCascade] = pd.concat([cascades[numCascade],childPortfCascade])
                        TtotalOrders = TtotalOrders[~TtotalOrders.isin(childPortfCascade)].dropna()

                        # add new stocks and their time affects, or extend time affects of previous stocks
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
    return cascades


def findStockOrderCascade(stockOrders, cascade, maxSep = 2):
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
    
def findPortfOrderCascade(TtotalOrders, portf, t0, maxSep = 1):
    """
    finds all subsequent activity traded by a portfolio (time threshold) (1 cascade only)
    input TtotalOrders, the portfolio, and minimum time
    Using Portf cascade is more logical than stock cascade
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

def findStockOrderCascades(stockOrders, cascades, minSep = 2):
    """
    inpurt filtered stockOrders table
    works on stockOrder tables from TtotalOrders, returns panda tables of each cascade
    returns all cascades
    FOR LOOKING AT INDIVIDUAL STOCK CASCADES NOV14
    """
    if len(stockOrders) > 0 :
        #Find seperations in trading times
        ToS = stockOrders['time'].unique()
        sep = np.diff(ToS)
        keySep = np.where(sep>minSep)[0]
        i0 = 0

        if len(cascades) > 0:
            n = max(cascades.keys())
        else:
            # first cascade
            n = 0

        if len(keySep)>0:
            for key in keySep:
                key += 1
                stockCascade = stockOrders[stockOrders['time']<=ToS[key]]
                stockCascade = stockCascade[stockCascade['time']>=ToS[i0]]
                cascades[n] = stockCascade
                i0 = key
                n += 1
                
    #cascades is a dictionary
    return cascades

    
def findPortfOrderCascades(TtotalOrders, portf, t0, maxSep = 1):
    """
    input TtotalOrders, the portfolio, and minimum time
    FOR REAL CASCADES Aug18
    Nov 14 edit to send multiple cascades
    """

    portfOrders = TtotalOrders[TtotalOrders['portfolio']==portf]
    if len(portfOrders)>0:
        ToS = portfOrders['time'].unique()
        sep = np.diff(ToS)

        keySep = np.where(sep>maxSep)[0]

        if len(keySep)>0:
            endtime = ToS[keySep[0]] #end of this cascade
        else:
            endtime = ToS[-1]
        portfCascade = portfOrders[portfOrders['time']<=endtime]
        return portfCascade
    else:
        return pd.DataFrame()


def cascadeAnalyzer(cascades, stockPool, t0 = 993, tf= 1992):
    """
    returns arrays of general cascade sizes (value), nrows, and duration for histogramming
    CALL FINDCASCADES FIRST FOR CAUSAL CASCADES
    """
    sizes = np.array([])
    nrows = np.array([])
    duration = np.array([])
    numCascade = len(cascades)
    for i in range(numCascade):
        casc = cascades[i]
        if casc['time'].min() > 993 and casc['time'].max() < 1992:
            size = 0
            for row in range(len(casc)):
                time = int(casc.iloc[row]['time'])
                stock = int(casc.iloc[row]['stock'])
                volume = int(casc.iloc[row]['order'])
                size += stockPool[stock][time]*abs(volume)
            dur = casc['time'].max()-casc['time'].min()
            if dur == 0:
                dur = 1
            duration = np.append(duration, dur)
            sizes = np.append(sizes, size)
            nrows = np.append(nrows, len(casc))
        # else:
            # print('passed cascade '+ str(i) + ' of size ' + str(len(cascades[i])))
            
    return sizes, nrows, duration

def stockOrderCascadeAnalyzer(Ttotalorders, stockPool, t0, tf, minSep ):    
    """
    analysis for stock order cascades seen in draft 8.#
    ALL IN ONE
    """
    cascades = {}
    for stock in range(len(stockPool)):
        cascades = findStockOrderCascades(Ttotalorders[Ttotalorders['stock']==stock], cascades = cascades, minSep = minSep)

    sizes, nrows, duration = cascadeAnalyzer(cascades, stockPool, t0=t0, tf=tf)

    return sizes, nrows, duration 