import numpy as numpy
import pandas as import pd
import matplotlib.pyplot as plt


def weightSharpeMovie(portf,TstockChars,TstockPool):
    """
    update to show +/- returns and color code each stock and average over time
    """
#     ovar = np.array([])
#     oalloc = np.array([])
#     omean = np.array([])
    ocashalloc = 0
    times = portf.valuedata.time.unique()
    stocks = portf.stocks
    ovar = np.zeros(len(stocks))
    oalloc = np.zeros(len(stocks))
    omean = np.zeros(len(stocks))
    ostd = np.zeros(len(stocks))
    
    for t in times[1::10]:
        alloc = np.array([])
        var = np.array([])
        mean = np.array([])
        std = np.array([])
        maxvar = max(TstockChars[TstockChars.stock.isin(stocks)]['var'])+0.5
        maxalloc = 1
        
        
        for s in stocks:
            weight = portf.weightdata[portf.weightdata.time==t]
            alloc = np.append(alloc, weight[weight.stock==s].weight[0]*TstockPool[s,t]/
                                 portf.valuedata[portf.valuedata.time==t].value[0])
            timeStockChars = TstockChars[TstockChars.time==t]
            var = np.append(var,timeStockChars[timeStockChars['stock']==s]['var'])
            mean = np.append(mean,timeStockChars[timeStockChars['stock']==s]['mean'])
            std = np.append(std,timeStockChars[timeStockChars['stock']==s]['std'])
            
        # include cash
        cashalloc = portf.valuedata[portf.valuedata.time==t].cash[0]/portf.valuedata[portf.valuedata.time==t].value[0]
        cashvar = 0

        plt.plot(cashvar, ocashalloc, 'o', color='lightgreen')
        plt.plot(cashvar, cashalloc, 'o', color='green', label='Cash')
        for i , s in enumerate(stocks):
#             plt.plot(omean[i]/ostd[i],oalloc[i],'o', color = 'lightgrey')
            plt.plot(mean[i]/std[i],alloc[i], 'o', label=s)
        plt.grid(True)
    #     plt.xlim(right=maxvar, left=0)
    #     plt.ylim(top=maxalloc,bottom=0)
        plt.xlabel("Stock Mean Return/Std")
        plt.ylabel("Weight Allocation")
        plt.legend()
        plt.title("Portf ID: " + str(portf.portfID) +" | Time: "+ str(t))
        plt.savefig("./results/7-9/"+str(t).zfill(6)+'_meanstd_tmp.png',dpi=250)
#         plt.show()
        plt.close()
        ovar = var
        ostd = std
        omean = mean
        oalloc = alloc 
        ocashalloc = cashalloc
    print("Done")
    print("mencoder 'mf://*meanstd_tmp.png' -mf type=png:fps=4 -ovc lavc -lavcopts vcodec=wmv2 -oac copy -o ./weightsharpe_"+portf.portfID+"_new2threshold_fixedhurst.mpg")
        
def weightVarMovie(portf,TstockChars,TstockPool):
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
        mean = np.array([])
        stocks = portf.stocks
        maxvar = max(TstockChars[TstockChars.stock.isin(stocks)]['var'])+0.5
        maxalloc = 1
        
        
        for s in stocks:
            weight = portf.weightdata[portf.weightdata.time==t]
            alloc = np.append(alloc, weight[weight.stock==s].weight[0]*TstockPool[s,t]/
                                 portf.valuedata[portf.valuedata.time==t].value[0])
            timeStockChars = TstockChars[TstockChars.time==t]
            var = np.append(var,timeStockChars[timeStockChars.stock==s]['var'])
            mean = np.append(mean,timeStockChars[timeStockChars['stock']==s]['mean'])
            
        # include cash
        cashalloc = portf.valuedata[portf.valuedata.time==t].cash[0]/portf.valuedata[portf.valuedata.time==t].value[0]
        cashvar = 0
        
        plt.plot(cashvar, ocashalloc, 'o', color='lightgreen')
        plt.plot(cashvar, cashalloc, 'o', color='green', label='Cash')
        for i , s in enumerate(stocks):
            if alloc[i] > 0:
                plt.plot(var[i],alloc[i], 'o', label=s)
            
        plt.grid(True)
    #     plt.xlim(right=maxvar, left=0)
    #     plt.ylim(top=maxalloc,bottom=0)
        plt.xlabel("Stock Variance")
        plt.ylabel("Weight Allocation")
        plt.legend()
        plt.title("Portf ID: " + str(portf.portfID) +" | Time: "+ str(t))
        plt.savefig("./results/7-9/"+str(t).zfill(6)+'_var_tmp.png',dpi=250)
        plt.show()
        ovar = var
        oalloc = alloc 
        ocashalloc = cashalloc
    print("Done")
    print("mencoder 'mf://*var_tmp.png' -mf type=png:fps=4 -ovc lavc -lavcopts vcodec=wmv2 -oac copy -o ./weightvar_"+portf.portfID+"_new2threshold_fixedhurst.mpg")

