import numpy as np 

stockPool = np.loadtxt('./fbm/fbm20_2_14_june25.txt')
#stockPool = stockPool[1:]


hurstPool = np.loadtxt('./fbm/fbm20_2_14_june25_hurstpool.txt')
#hurstPool = hurstPool[1:]

window = 900