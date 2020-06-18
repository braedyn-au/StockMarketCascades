import numpy as np
import pandas as pd 
from library import config

stockPool = config.stockPool

def sharpe(alloc, stocks, vol, ti, tf):
    """
    remake of sharpe calculation following 
    https://www.mlq.ai/python-for-finance-portfolio-optimization/#h1sjvcte25p1r8or1e1ngd82h2r8ha1
    uses allocation percentage instead of weights
    """
    #print(np.shape(stockPool))
    Rp = 0
    var = 0
    Rf = 0.010
    for i,j in enumerate(stocks): 
        stepReturn = 100*np.diff(stockPool[j][ti:tf])/stockPool[j][ti:tf-1]
        Rp += alloc[i]*vol*np.mean(stepReturn)
        var += alloc[i]*alloc[i]*vol*vol*np.var(stepReturn)
    stdp = np.sqrt(var)
    
    #print(Rp,stdp)
    if stdp == 0:
        print("dividebyzero")
    return -(Rp-Rf)/stdp

def characterize(tmin = 992, tmax = 8192, window=config.window):# stockPool=stockPool):
    """
    returns info of the stocks leading up to the optimization,
    such as variance of each stock and the gap between highest and lowest

    not efficient, better to just have a global stockChars df where I lookup stocks corresponding to each portfolio
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