import numpy as np
from scipy.optimize import minimize
import pandas as pd
import random 
import string
from libraryMP.utils import sharpe, characterize, sigmoid
from libraryMP import config
from fbm.fbm import fbm
# import matlab.engine

# eng = matlab.engine.start_matlab()

stockPool = np.copy(config._stockPool)
hurstPool = np.copy(config._hurstPool)
changePrice = config.changePrice
tinit = config.tinit


class portfolio:
    
    def __init__(self,name,size,volume, stocks):
        self.volume = volume
        self.stocks = stocks
        self.alloc = dict.fromkeys(self.stocks,1/size) 
        self.alloc['cash'] = 0 # cash  
        self.weights = dict.fromkeys(self.stocks,1) # put in dictionary for easy change
        self.orders = np.zeros(size)
        self.portfID = str(name)
        self.sharpeOpt = np.asarray([])
        self.sharpeReal = np.asarray([])
        self.initAlloc = np.asarray([])
        self.sharpeNonOpt = np.asarray([])
        self.stockChars = pd.DataFrame()
        self.threshold = 0
        self.maxSharpe = 0
        self.weightdata = pd.DataFrame()
        self.valuedata = pd.DataFrame()
        self.value = np.asarray([self.volume])
        self.cash = np.asarray([])
    
    def optimize(self, t = tinit,first=False, window=config.window ):#, stockPool=stockPool):
        """
        initialize weights based on datapoints before the 5 day week
        """
        ti = t-window
        
        def check_sum(alloc):
            return np.sum(alloc)-1
        cons = ({'type':'eq','fun':check_sum})
        
        bounds = []
        for i in range(len(self.stocks)+1):
            bounds.append((0,1))

        # check alloc percent equals 1
        try:
            assert round(np.sum(list(self.alloc.values()))) == 1
        except:
            print('/n')
            print("PortfID: ", self.portfID)
            print('Alloc: ',self.alloc)
            print('Weight: ', self.weights)
            print("Init Value: ", self.volume)
            print('Value: ', self.value[-1])
            print('Sharpe: ', self.sharpeReal)
            print('Sharpe Opt: ', self.sharpeOpt)
            print("Non 100 allocation")
            raise

        opt= (minimize(sharpe, 
                      list(self.alloc.values()), 
                      args=(stockPool,self.stocks,self.value[-1],ti,t), 
                      method='SLSQP', 
                      bounds=bounds,
                      constraints=cons)['x'])

        if first:
            self.initAlloc = opt
            for i,j in enumerate(self.stocks):
                self.weights[j] = np.floor(opt[i]*self.volume/stockPool[j][tinit])
                self.alloc[j] = self.weights[j]*stockPool[j][tinit]/self.volume
            self.alloc['cash'] = 1 - np.sum(list(self.alloc.values())[:-1])
            self.cash = np.append(self.cash,self.alloc['cash']*self.volume)
            print('Cash: ', self.alloc['cash'], "| $",self.cash[-1])
            print("Optimal Sharpe: ", -sharpe(opt,stockPool,self.stocks,self.volume,ti,t))
            print("Initial Sharpe: ", -sharpe(list(self.alloc.values()),stockPool,self.stocks,self.volume,ti,t))
    
        self.sharpeNonOpt = np.append(self.sharpeNonOpt,-sharpe(self.initAlloc,stockPool,self.stocks,self.value[-1],ti,t))
        self.sharpeOpt = np.append(self.sharpeOpt,-sharpe(opt,stockPool,self.stocks,self.value[-1],ti,t))
        self.sharpeReal = np.append(self.sharpeReal,-sharpe(list(self.alloc.values()),stockPool,self.stocks,self.value[-1],ti,t))
            
            
        return opt
        
    def thresholdOrder(self, time, window=config.window):
        self.updateWeightData(time)
        ti = time-window
        opt = self.optimize(t=time)
        new = -sharpe(opt,stockPool,self.stocks,self.value[-1],ti,time)
        
        pthresh = sigmoid(new,self.threshold)
        puni = np.random.rand()
        print("rebalance prob: ", pthresh)
        print("roll: ", puni)
        if puni < pthresh:
            print('order sent')
            orderList = self.order(time,opt=opt)
        else:
            blank = np.zeros(len(self.stocks))
            orderList = pd.DataFrame()
        return orderList

    def order(self, time, opt=None, changePrice = changePrice):#, stockPool = stockPool):
        """
        calls optimize to find opt alloc
        returns the orders to be added to the broker dataframe
        immediately adjusts weights for sold stocks
        """

        if opt is None:
            # Means no thresholding
            opt = self.optimize(t=time)
            self.updateWeightData(time)

        optweights = []
        for i,j in enumerate(self.stocks):
            optweights.append(np.floor(opt[i]*self.value[-1]/stockPool[j][time]))
        
        self.orders = np.asarray(optweights) - list(self.weights.values())
        orderList = pd.DataFrame({'time':time, "portfolio":self.portfID,"stock":self.stocks, "order": self.orders})
    
        # update weights that have been sent off and portfolio value
        i = 0
        fvalue = 0
        fcash = self.cash[-1]
        for stock,weight in self.weights.items():
            self.weights[stock] = weight + self.orders[i]
            if self.orders[i]<0:
                #SELL
                fcash += -self.orders[i]*stockPool[stock][time]
            elif self.orders[i]>0:
                #BUY
                fcash += -self.orders[i]*stockPool[stock][time]
        
            # Update portfolio net value
            fvalue += self.weights[stock]*stockPool[stock][time]

            if changePrice and self.orders[i]!=0:
                hurstChange_random(stock,time,volume=self.orders[i])

            i+=1

        fvalue += fcash
        self.value = np.append(self.value,fvalue)            
        self.cash = np.append(self.cash, fcash)        
        # renormalize alloc based on new portfolio value
        for stock,weight in self.weights.items():
            self.alloc[stock] = self.weights[stock]*stockPool[stock][time]/self.value[-1] # i think this works
        
        # get cash percentage by normalizing weight
        self.alloc['cash'] = fcash/self.value[-1]

        if self.alloc['cash'] < 0 :
            print(self.portfID, " cash allocation negative at ", self.alloc['cash'])

        return orderList[orderList.order!=0]
    
    def buy(self,stock,time,volume, changePrice=changePrice):
        """
        LEGACY
        adjust recently bought stocks, volume of stock bought
        merged from broker_funcs with order for instant buy/sell 
        """
        self.weights[stock] = self.weights[stock] + volume
        ivalue = self.value[-1]
        self.value = np.append(self.value, ivalue + volume*stockPool[stock][time])
        # self.alloc[stock] = self.weights[stock]*stockPool[stock][time]/self.value[-1] needs to be done all at once
        if changePrice:
            hurstChange_random(stock,time,volume=volume)


    def updateWeightData(self, time):
        """
        update the weightData and valueData tables to current weights of each stock
        """

        fvalue = 0
        for stock, weight in self.weights.items():
            self.weightdata = pd.concat([self.weightdata,pd.DataFrame({'ID':self.portfID,'time':[time-1], 'stock':stock,'weight':self.weights[stock]})])
            fvalue += self.weights[stock]*stockPool[stock][time]
        fvalue += self.cash[-1]
        self.value = np.append(self.value,fvalue) 
        self.valuedata = pd.concat([self.valuedata, pd.DataFrame({'ID':self.portfID, 'time': [time-1], 'value': self.value[-1], 'cash': self.cash[-1]})])
        # TODO: calculate actual current value

    def reset(self, t = tinit, ptile=70):
        """
        only to be used once after dry run
        reset alloc and time, find the sharpe ratio threshold for sigmoid
        """
        #global stockPool, hurstPool

        size = len(self.stocks)
        self.alloc = dict.fromkeys(self.stocks,1/size) #init with a sharpe function,
        self.alloc['cash'] = 0
        self.initAlloc = np.asarray([])
        self.weights = dict.fromkeys(self.stocks,1) # put in dictionary for easy change
        self.orders = np.zeros(size)
        percentile = np.percentile(self.sharpeReal,ptile)
        self.threshold = percentile # just the ptile
        self.sharpeOpt = np.asarray([])
        self.sharpeReal = np.asarray([])
        self.sharpeNonOpt = np.asarray([])
        self.cash = np.asarray([])
        self.value = np.asarray([self.volume])
        self.weightdata = pd.DataFrame()
        self.valuedata = pd.DataFrame()

        
        self.optimize(first=True)
        print('reset!')
        print('threshold: ',self.threshold)
        print("_____")

        resetStocks()
        checkReset()

        
def portfGen(stockPool=stockPool, n=config.nportfs, sizeMin=config.minPortfSize, sizeMax=config.maxPortfSize, overlapMin = config.overlapMin, overlapMax=config.overlapMax):
    """
    updated to include overlap function june 25
    """
    stocks = np.arange(np.shape(stockPool)[0]) #check if 0 or 1
    print(stocks)
    traderIDs = {}
    
    def randString(length = 5):
        letters = string.ascii_lowercase
        return ''.join(random.sample(letters,length))
    
    # for i in range(n):
    #     name = randString()
    #     print(name)
    #     while name in traderIDs:
    #         name = randString()
    #     vol = 10**np.random.randint(3,6)
    #     traderIDs[name] = portfolio(name,np.random.randint(8,12),vol)
    #     traderIDs[name].optimize(first=True)
    indx = 0
    for portfs in range(n):
        
        window = np.random.randint(sizeMin,sizeMax)
        overlap = np.random.randint(overlapMin,overlapMax)
        vol = 10**(np.random.randint(4,6))

        startpos = indx % len(stocks)
        name = randString()
        print(name)
        while name in traderIDs:
            name = randString()
    #     print(startpos)
    #     print(stocks)
    #     print(startpos, ":", startpos+window)
        if startpos+window >= len(stocks):
            print('overflow ', startpos+window)
            stocks2 = np.concatenate([stocks,stocks])
            tstocks = np.copy(stocks2[startpos:startpos+window])
            np.random.shuffle(stocks)
            print("shuffled")
        else:
            tstocks = np.copy(stocks[startpos:startpos+window])
        traderIDs[name] = portfolio(name, window, vol, tstocks)
        traderIDs[name].optimize(first=True)
        indx += window - overlap

    
    return traderIDs
    
def uniquePortfGen(n=5, availStocks = np.shape(stockPool)[0]):
    """
    no overlap
    """
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

def stockChars():
    return stockPool,hurstPool
    
def resetStocks():
    """
    reset global variables
    """
    global stockPool, hurstPool
    stockPool = np.copy(config._stockPool)
    hurstPool = np.copy(config._hurstPool)

def checkReset():
    """
    make sure reset worked
    """
    if not np.mean(stockPool) == np.mean(config._stockPool) and np.mean(hurstPool) == np.mean(_config.stockPool):
        raise Exception("reset not work")

def hurstChange_updown(stock, time, volume, proportion = 10000):
    """
    Updates the stockPool and hurstPool as price change
    Sell => decrease Hurst (more volatile)
    Buy => increase Hurst (less volatile)
    """
    global stockPool, hurstPool 

    increment = volume/proportion
    h0 = hurstPool[stock][time]
    numberNewPrices = len(stockPool[stock+1][time:])
    p0 = stockPool[stock][time]
    
    if volume<0:
        h1 = h0-increment
        if h1<0.4:
            h1=0.4
        hurstPool[stock][time+1:] = h1
    else:
        h1 = h0+increment
        if h1>0.8:
            h1=0.8
        hurstPool[stock][time+1:] = h1

    # fbmNew = fbm(h1,2**14,2**14)
    # fbmNew = abs(fbmNew[:numberNewPrices]+p0)
    print('stock ', stock, ' original H', h0, ' to ', h1)
    # stockPool[stock][time:]=fbmNew

def hurstChange_random(stock, time, volume, proportion = 1000):
    """
    Updates the stockPool and hurstPool as price change
    Random +/- increment to Hurst
    """
    global stockPool, hurstPool 

    increment = np.random.choice(np.linspace(-volume/proportion,volume/proportion,7))

    if increment != 0:
        # print(volume)
        # print(increment)
        h0 = hurstPool[stock][time+1]
        # numberNewPrices = len(stockPool[stock][time+1:])
        # p0 = stockPool[stock][time]
        
        h1 = h0+increment
        if h1<0.45:
            h1=0.2
        elif h1>0.75:
            h1=0.75
        
        hurstPool[stock][time+1:] = h1
        # fbmNew = fbm(h1,2**14,2**14)
        # fbmNew = abs(fbmNew[:numberNewPrices]+p0)
        # if stock == 0:
        #     print('stock ', stock, ' original H', h0, ' to ', h1, " | ", h1-h0)
        # stockPool[stock][time+1:]=fbmNew

def priceChange(time, changePrice=changePrice):
    """
    takes the newest updated hurst index and pushes new price
    added july 27
    """
    global stockPool, hurstPool
    
    if changePrice == True:
        for stock in range(len(stockPool)):
            h0 = hurstPool[stock][time]
            h1 = hurstPool[stock][time+1]
            if h1!=h0:
                print("stock: ", stock, " | change H: ",h1-h0)
                numberNewPrices = len(stockPool[stock][time+1:])
                p0 = stockPool[stock][time]
                fbmNew = fbm(h1, 2**14,2**14)
                fbmNew = abs(fbmNew[:numberNewPrices]+p0)
                stockPool[stock][time+1:]=fbmNew
            


