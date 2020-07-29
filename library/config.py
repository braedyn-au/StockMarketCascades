import numpy as np 

window = 500
changePrice = True
nportfs = 501
minPortfSize = 9
maxPortfSize = 12
overlapMin = 7
overlapMax = 9
tinit = 992
simsteps = 1000
threshold = 100
sigmoid = 60
leak = 0.9
stockPool = './fbm/fbm100_2_14_2_5.txt'
hurstPool = './fbm/fbm100_2_14_2_5_hurstpool.txt'
_stockPool = np.loadtxt(stockPool)
_hurstPool = np.loadtxt(hurstPool)

config = {k:v for k, v in locals().items() if not k.startswith('_')}
