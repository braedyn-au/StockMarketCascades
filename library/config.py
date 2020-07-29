import numpy as np 

window = 500
changePrice = True
nportfs = 1000
minPortfSize = 9
maxPortfSize = 12
overlapMin = 8
overlapMax = 10
tinit = 992
threshold = 60
stockPool = './fbm/fbm100_2_14_july6.txt'
hurstPool = './fbm/fbm100_2_14_july6_hurstpool.txt'
_stockPool = np.loadtxt(stockPool)
_hurstPool = np.loadtxt(hurstPool)

config = {k:v for k, v in locals().items() if not k.startswith('_')}