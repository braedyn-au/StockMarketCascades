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

def sigmoid(x, x0, k = 100):
    z = np.exp(-k*(x-x0))
    p = 1/(1+z)
    return p

# def fbm(H,n=(2**14),T=(2**14)):
#     """
#     H - hurst index in float format
#     n - number of timesteps as multiple of 2 in float format
#     T - total time can keep 1?? float
    
#     calls the matlab function from wikipedia
#     ensure import matlab.engine
#     Zdravko Botev (2020). Fractional Brownian motion generator 
#     (https://www.mathworks.com/matlabcentral/fileexchange/38935-fractional-brownian-motion-generator)
#     Kroese, D. P., & Botev, Z. I. (2015). Spatial Process Simulation.
#     In Stochastic Geometry, Spatial Statistics and Random Fields(pp. 369-404)
#     Springer International Publishing, DOI: 10.1007/978-3-319-10064-7_12
#     """

#     T = float(T)
#     n = float(n)
#     H = float(H)
#     print(type(H))
#     print(H)
    
#     eng = matlab.engine.start_matlab()
#     a = eng.fbm1d(H,n,T)
#     fbm = np.asarray([])
#     for i in range(len(a)):
#         fbm = np.concatenate((fbm,np.asarray(a[i])))

#     #plt.plot(np.linspace(0,1,int(n+1)),fbm)
    
#     return fbm







