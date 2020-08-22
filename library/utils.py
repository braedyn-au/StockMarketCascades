import numpy as np
import pandas as pd 
from library import config
# import matlab.engine

# eng = matlab.engine.start_matlab()
#stockPool = config.stockPool

def sharpe( alloc, stockPool, stocks, vol, ti, tf):
    """
    remake of sharpe calculation following 
    https://www.mlq.ai/python-for-finance-portfolio-optimization/#h1sjvcte25p1r8or1e1ngd82h2r8ha1
    uses allocation percentage instead of weights
    """
    #print(np.shape(stockPool))

    # if np.sum(alloc[:-1])==0:
    #     print("all cash")
    #     return 0
    # else:
    Rp = 0
    var = 0
    Rf = 0.010 * alloc[-1]*vol # cash return
    for i,j in enumerate(stocks): 
        stepReturn = 100*np.diff(stockPool[j][ti:tf])/stockPool[j][ti:tf-1]
        Rp += alloc[i]*vol*np.mean(stepReturn)
        var += alloc[i]*alloc[i]*vol*vol*np.var(stepReturn)
    stdp = np.sqrt(var)
    
    return -(Rp-Rf)/stdp

def characterize(stockPool, tmin = 992, tmax = 8192, window=config.window):# stockPool=stockPool):
    """
    returns info of the stocks leading up to the optimization,
    such as variance of each stock and the gap between highest and lowest

    not efficient, better to just have a global stockChars df where I lookup stocks in the stockPool corresponding to each portfolio
    ***moved to utils
    """
    stockChars = pd.DataFrame()        
    
    for tf in range(tmin, tmax):
        ti = tf-window
        for stock in range(np.shape(stockPool)[0]):
            stepReturn = 100*np.diff(stockPool[stock][ti:tf])/stockPool[stock][ti:tf-1]
            var = np.var(stepReturn)
            std = np.sqrt(var)
            mean = np.mean(stepReturn)
            char = pd.DataFrame({'time':[tf],'stock':stock,'mean':mean,'var':var,'std':std})
            stockChars = pd.concat([stockChars,char])
    return stockChars

def sigmoid(x, x0, k = 50):
    z = np.exp(-k*(x-x0))
    p = 1/(1+z)
    return p

def cosineSim(portf1, portf2): #change from .stocks array to the weightdata arrays from objects
    """
    takes two portfolio objects and finds overlap between them

    """
    overlapWeights = 0
    for i in portf1.stocks:
        if i in portf2.weights: # find the weights in portf1 and portf2
            overlapWeights += (portf1.weights[i] + portf2.weights[i]) # actual weight calculation
    
    portf1weights = np.asarray(list(portf1.weights.values()))
    portf2weights = np.asarray(list(portf2.weights.values()))
    portf1Norm = np.linalg.norm(portf1weights)
    portf2Norm = np.linalg.norm(portf2weights)
    
    return overlapWeights/(portf1Norm*portf2Norm)

def cosineSimP(time, portf1, portf2): #change from .stocks array to the weightdata arrays from objects
    """
    takes two portfolio objects and finds overlap between them at a certain time

    """
    overlapWeights = 0
    ptime1 = portf1.weightdata[portf1.weightdata['time']==time]
    ptime2 = portf2.weightdata[portf2.weightdata['time']==time]
    for i in portf1.stocks:
        if i in portf2.weights: # find the weights in portf1 and portf2
            pweight1 = int(ptime1[ptime1['stock']==i].weight)
            pweight2 = int(ptime2[ptime2['stock']==i].weight)
            overlapWeights += (pweight1 * pweight2) # actual weight calculation

    portf1weights = (ptime1.weight.values)
    portf2weights = (ptime2.weight.values)
    portf1Norm = np.linalg.norm(portf1weights)
    portf2Norm = np.linalg.norm(portf2weights)
    
    if portf1.portfID == portf2.portfID:
        assert round(overlapWeights/(portf1Norm*portf2Norm)) == 1
    
    return overlapWeights/(portf1Norm*portf2Norm)

def cosineSimP_cash(time, portf1, portf2): #change from .stocks array to the weightdata arrays from objects
    """
    takes two portfolio objects and finds overlap between them including cash assets

    """
    overlapWeights = 0
    ptime1 = portf1.weightdata[portf1.weightdata['time']==time]
    ptime2 = portf2.weightdata[portf2.weightdata['time']==time]
    pcash1 = portf1.valuedata[portf1.valuedata['time']==time].cash
    pcash2 = portf2.valuedata[portf2.valuedata['time']==time].cash
    for i in portf1.stocks:
        if i in portf2.weights: # find the weights in portf1 and portf2
            pweight1 = int(ptime1[ptime1['stock']==i].weight)
            pweight2 = int(ptime2[ptime2['stock']==i].weight)
            overlapWeights += (pweight1 * pweight2) # actual weight calculation
    overlapWeights += pcash1*pcash2
    
    portf1weights = np.append((ptime1.weight.values,pcash1))
    portf2weights = np.append((ptime2.weight.values,pcash2))
    portf1Norm = np.linalg.norm(portf1weights)
    portf2Norm = np.linalg.norm(portf2weights)
    
    if portf1.portfID == portf2.portfID:
        assert round(overlapWeights/(portf1Norm*portf2Norm)) == 1
    
    return overlapWeights/(portf1Norm*portf2Norm)

def findStockCascades(stockTrans, maxSep = 10):
    """
    works on stockTransactions tables
    """
    
    if len(stockTrans) > 0 :
        ToS = stockTrans.ToS.unique()
        sep = np.diff(ToS)
        keySep = np.where(sep>maxSep)[0]

        i0 = 0
        cascadeSizes = np.asarray([])
        for key in keySep:
            key += 1
            times = ToS[i0:key]
            cascadeSize = len(stockTrans[stockTrans.ToS.isin(times)])
            cascadeSizes = np.append(cascadeSizes,cascadeSize)
            i0 = key

        return cascadeSizes
    else:
        return []

def weightvarMovie(portf,TstockChars,TstockPool):
    """
    aug 21 update to show +/- returns and color code each stock and average over time
    """
    ovar = np.array([])
    oalloc = np.array([])
    ocashalloc = 0
    times = portf.valuedata.time.unique()
    
    for t in times[1::10]:
        alloc = np.array([])
        var = np.array([])
        stocks = portf.stocks
        maxvar = max(TstockChars[TstockChars.stock.isin(stocks)]['var'])+0.5
        maxalloc = 1
        
        
        for s in stocks:
            weight = portf.weightdata[portf.weightdata.time==t]
            alloc = np.append(alloc, weight[weight.stock==s].weight[0]*TstockPool[s,t]/
                                 portf.valuedata[portf.valuedata.time==t].value[0])
            timeStockChars = TstockChars[TstockChars.time==t]
            var = np.append(var,timeStockChars[timeStockChars.stock==s]['var'])
            
        # include cash
        cashalloc = portf.valuedata[portf.valuedata.time==t].cash[0]/portf.valuedata[portf.valuedata.time==t].value[0]
        cashvar = 0
        
        plt.plot(cashvar, ocashalloc, 'o', color='lightgreen')
        plt.plot(cashvar, cashalloc, 'o', color='green', label='Cash')
        plt.plot(ovar,oalloc,'o', color = 'lightblue')
        plt.plot(var,alloc, 'o')
        plt.grid(True)
    #     plt.xlim(right=maxvar, left=0)
    #     plt.ylim(top=maxalloc,bottom=0)
        plt.xlabel("Stock Variance")
        plt.ylabel("Weight Allocation")
        plt.legend()
        plt.title("Portf ID: " + str(portf.portfID) +" | Time: "+ str(t))
        plt.savefig("./results/50_traders_change_price/w_valuetable/"+str(t).zfill(6)+'wv_tmp.png',dpi=250)
        plt.show()
        ovar = var
        oalloc = alloc 
        ocashalloc = cashalloc
    print("Done")
    print("mencoder 'mf://*wv_tmp.png' -mf type=png:fps=4 -ovc lavc -lavcopts vcodec=wmv2 -oac copy -o ./weightvar_"+portf.portfID+"_pricechange_thresholding.mpg")







