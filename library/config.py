import numpy as np 

stockPool = np.loadtxt('./fbm/fbm20+min+100.txt')
stockPool = stockPool[1:]

window = 900