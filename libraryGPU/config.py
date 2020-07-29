import numpy as np 
import cupy
window = 500
changePrice = True
nportfs = 50
minPortfSize = 9
maxPortfSize = 12
overlapMin = 8
overlapMax = 10
tinit = 992
stockPool = './fbm/fbm100_2_14_july6.txt'
hurstPool = './fbm/fbm100_2_14_july6_hurstpool.txt'
_stockPool = cupy.asarray(np.loadtxt(stockPool))
_hurstPool = cupy.asarray(np.loadtxt(hurstPool))

config = {k:v for k, v in locals().items() if not k.startswith('_')}