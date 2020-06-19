import numpy as np
from scipy.optimize import minimize
import pandas as pd
import random 
import string
from library.utils import sharpe
from library import config

stockPool = config.stockPool

class portfolio:
    
    def __init__(self,name,size,volume):
        self.volume = volume
        self.stocks = np.random.choice(np.arange(20), size=size, replace=False) # REPLACE
        self.alloc = dict.fromkeys(self.stocks,1/size) #init with a sharpe function,
        self.weights = dict.fromkeys(self.stocks,1) # put in dictionary for easy change
        self.orders = np.zeros(size)
        self.portfID = str(name)
        self.sharpe = np.asarray([])
        self.initAlloc = np.asarray([])
        self.sharpeNonOpt = np.asarray([])
        self.stockChars = pd.DataFrame()
        self.threshold = 0
    
    def optimize(self, t = 992,first=False, window=config.window, ):#, stockPool=stockPool):
        """
        initialize weights based on datapoints before the 5 day week
        """
        ti = t-window
        
        def check_sum(alloc):
            return np.sum(alloc)-1
        cons = ({'type':'eq','fun':check_sum})
        
        bounds = []
        for i in range(len(self.stocks)):
            bounds.append((0,1))
        
        opt= (minimize(sharpe, 
                      list(self.alloc.values()), 
                      args=(self.stocks,self.volume,ti,t), 
                      method='SLSQP', 
                      bounds=bounds,
                      constraints=cons)['x'])
        self.sharpe = np.append(self.sharpe,-sharpe(opt,self.stocks,self.volume,ti,t))
        #print('opt: ',opt)
        #print('sum opt: ', np.sum(opt))
        # update self.alloc
        if first:
            self.initAlloc = opt
            self.sharpeNonOpt = np.append(self.sharpeNonOpt,-sharpe(self.initAlloc,self.stocks,self.volume,ti,t))
            for i,j in enumerate(self.stocks):
                self.weights[j] = round(opt[i]*self.volume/stockPool[j][992])
                self.alloc[j] = opt[i]
            print(-sharpe(list(self.alloc.values()),self.stocks,self.volume,ti,t))
        else:
            self.sharpeNonOpt = np.append(self.sharpeNonOpt,-sharpe(self.initAlloc,self.stocks,self.volume,ti,t))
            #print(self.sharpeNonOpt)
            for i,j in enumerate(self.stocks):
                self.alloc[j] = opt[i]
            return opt
    
    def characterize(self, tf, window=config.window):# stockPool=stockPool):
        """
        ***moved to utils
        returns info of the stocks leading up to the optimization,
        such as variance of each stock and the gap between highest and lowest

        not efficient, better to just have a global stockChars df where I lookup stocks corresponding to each portfolio
        """
        ti = tf-window        
        
        for stock in self.stocks:
            stepReturn = 100*np.diff(stockPool[stock][ti:tf])/stockPool[stock][ti:tf-1]
            var = np.var(stepReturn)
            std = np.sqrt(var)
            mean = np.mean(stepReturn)
            char = pd.DataFrame({'time':[tf], "portfolio":self.portfID,'stock':stock,'mean':mean,'var':var,'std':std})
            self.stockChars = pd.concat([self.stockChars,char])
        

    def order(self, time, threshold=False):#, stockPool = stockPool):
        """
        calls optimize to find opt alloc
        returns the orders to be added to the broker dataframe
        immediately adjusts weights for sold stocks
        """
        opt = self.optimize(t=time)
        if threshold: ###

        optweights = []
        for i,j in enumerate(self.stocks):
            optweights.append(round(opt[i]*self.volume/stockPool[j][time]))
        
        self.orders = np.asarray(optweights) - list(self.weights.values())
        orderList = pd.DataFrame({'time':time, "portfolio":self.portfID,"stock":self.stocks, "order": self.orders})
        
        # update weights that have been sent off
        i = 0
        for stock,weight in self.weights.items():
            if self.orders[i]<0:
                self.weights[stock] = weight + self.orders[i]
            i+=1
        return orderList
    
    def buy(self,stock,volume):
        """
        adjust recently bought stocks
        """
        
        #print(self.weights)
        self.weights[stock] = self.weights[stock] + volume
        #print(self.weights)
        
def portfGen(n=5):
    """
    update to include overlap function
    """

    traderIDs = {}
    
    def randString(length = 5):
        letters = string.ascii_lowercase
        return ''.join(random.sample(letters,length))
    
    for i in range(n):
        name = randString()
        print(name)
        while name in traderIDs:
            name = randString()
        vol = 10**np.random.randint(3,6)
        traderIDs[name] = portfolio(name,np.random.randint(8,12),vol)
        traderIDs[name].optimize(first=True)
        
    return traderIDs
    
def uniquePortfGen(n=5, availStocks = np.shape(stockPool)[0]):
    traderIDs = {}
    
    def randString(length = 5):
        letters = string.ascii_lowercase
        return ''.join(random.sample(letters,length))
    dist = np.arange(availStocks).reshape((5,4))
    
    for i in range(n):
        name = randString()
        print(name)
        while name in traderIDs:
            name = randString()
        vol = 10**6
        traderIDs[name] = portfolio(name,int(availStocks/n),vol)
        traderIDs[name].stocks = dist[i]
        traderIDs[name].alloc = dict.fromkeys(traderIDs[name].stocks,int(availStocks/n)) #init with a sharpe function,
        traderIDs[name].weights = dict.fromkeys(traderIDs[name].stocks,1) # put in dictionary for easy change
        traderIDs[name].optimize(first=True)
        print(traderIDs[name].stocks)
        print(traderIDs[name].weights)
        print(traderIDs[name].alloc)
        
    return traderIDs
    