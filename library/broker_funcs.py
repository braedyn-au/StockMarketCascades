import numpy as np 
import pandas as pd 
from library import config

stockPool = config.stockPool

def brokerage(traderIDs, time, broker,totalOrders):
    """
    generates the brokker list which splits
    into potMatch (sales) and notMatch (back to broker)
    
    can be used to look at the total submitted orders...
    """
    for key,portf in traderIDs.items():
        #portf.characterize(time) moved to utils and perform on whole stockPool
        orderList = portf.order(time=time)
        broker = pd.concat([broker,orderList])
        totalOrders = pd.concat([totalOrders,orderList[orderList.order!=0]])
    #print(broker[broker.order!=0])

    return broker[broker.order!=0], totalOrders #remove all null orders

def thresholdBrokerage(traderIDs,time,broker,totalOrders):
    """
    resets the portfolios and runs thresholdOrder
    """
    for key,portf in traderIDs.items():
        orderList = portf.thresholdOrder(time=time)
        broker = pd.concat([broker,orderList])
        totalOrders = pd.concat([totalOrders,orderList[orderList.order!=0]])
    
    return broker[broker.order!=0], totalOrders


def match(traderIDs, broker, transactions):
    """
    does potMatch and notMatch
    """
    potMatch = broker[broker.groupby('stock').stock.transform(len) > 1] # removes all single orders
    notMatch = broker[broker.groupby('stock').stock.transform(len) == 1]
    
    for stock in (potMatch.stock.unique()): 
        # go through each stock and try to fill orders
        stockSearch = (potMatch[potMatch.stock == stock])
        buy = stockSearch[stockSearch.order > 0]
        sell = stockSearch[stockSearch.order < 0]
        if len(buy) == 0:
            notMatch = pd.concat([notMatch,sell])
        elif len(sell) == 0:
            notMatch = pd.concat([notMatch,buy])
        else:
            buy = buy.sort_values(by=["time",'order'],ascending=[True,False])
            sell = sell.sort_values(by=["time",'order'],ascending=[True,True])

            #print("BUY")
            #print(buy)
            #print("SELL")
            #print(sell)

            while len(buy) != 0 and len(sell) != 0:
                # uncomment prints to check logic
                # may want to include cancelling orders
                
                buy.reset_index(inplace=True,drop=True)
                sell.reset_index(inplace=True,drop=True)
                # iterate until no more matches
                sID = 0 # always look at the most important selling order first 
                #print(buy)
                #print(sell)
                sVol = sell.iloc[sID].order #selling amount

                #else:
                # algorithm that matches s with a b and updates sell and buy rows, take min and subtract until 0
                bMatch = buy.iloc[0]
                bVol = bMatch.order

                Vol = min(abs(sVol),abs(bVol))
                ToS = max(sell.iloc[sID].time, bMatch.time)

                sale = pd.DataFrame({"ToS":[ToS], "stock":stock, "seller": sell.iloc[sID].portfolio, 
                                     "buyer": bMatch.portfolio, "volume": Vol, 
                                     "tradeID": str(ToS)+'|'+str(stock)+str(sell.iloc[sID].portfolio)
                                    +str(bMatch.portfolio)+str(-sVol)})

                transactions = pd.concat([transactions,sale])


                sell.set_value(sID, 'order', sell.iloc[sID].order + Vol)
                buy.set_value(0, 'order', bVol - Vol)


                #print("sale")

                #update buyer weights
                #print(bMatch.portfolio)
                #print(traderIDs[bMatch.portfolio])
                buyer = traderIDs[bMatch.portfolio]
                buyer.buy(stock,Vol)


                buy = buy[buy.order!=0]
                sell = sell[sell.order!=0]
                #print(buy)
                #print(sell)
                #print("______________ \n")

            if len(buy)==0:
                notMatch = pd.concat([notMatch,sell])
            else:
                notMatch = pd.concat([notMatch,buy])


    broker = notMatch
    return broker, transactions

def instantMatch(traderIDs, broker, transactions):
    """
    instantly matches orders with a seller/buyer outside
    """
    
    broker = broker.sort_values(by='time', ascending=True)      
    broker.reset_index(inplace=True,drop=True)
    
    for sID in range(len(broker)):
    # iterate through all broker orders

        Vol = broker.iloc[sID].order #selling amount
        ToS = broker.iloc[sID].time
        stock = broker.iloc[sID].stock
        trader = traderIDs[broker.iloc[sID].portfolio]

        if Vol > 0:
            trader.buy(stock=stock,time=ToS,volume=Vol)
            sale = pd.DataFrame({"ToS":[ToS], "stock":stock, "seller": 'world', 
                             "buyer": trader.portfID, "volume": Vol, 
                             "tradeID": str(ToS)+'|'+str(stock)+str('world')
                            +str(trader.portfID)+str(Vol)})
        else:
            sale = pd.DataFrame({"ToS":[ToS], "stock":stock, "seller": trader.portfID, 
                             "buyer": 'world', "volume": Vol, 
                             "tradeID": str(ToS)+'|'+str(stock)+str('world')
                            +str(trader.portfID)+str(Vol)})

        transactions = pd.concat([transactions,sale])


        broker.set_value(sID, 'order', 0)
    broker = broker[broker.order!=0]
    
    if len(broker) != 0:
        print('OOPS len(broker) != 0')

        #print("sale")

        #update buyer weights
        #print(bMatch.portfolio)
        #print(traderIDs[bMatch.portfolio])


    
    return broker, transactions


