print("Importing MATLAB")

import matlab.engine
import numpy as np
eng = matlab.engine.start_matlab()
eng.cd(r'fbm', nargout=0)

def fbm(H,n=(2**14),T=(2**14)):
    """
    H - hurst index in float format
    n - number of timesteps as multiple of 2 in float format
    T - total time can keep 1?? float
    
    calls the matlab function from wikipedia
    ensure import matlab.engine
    Zdravko Botev (2020). Fractional Brownian motion generator 
    (https://www.mathworks.com/matlabcentral/fileexchange/38935-fractional-brownian-motion-generator)
    Kroese, D. P., & Botev, Z. I. (2015). Spatial Process Simulation.
    In Stochastic Geometry, Spatial Statistics and Random Fields(pp. 369-404)
    Springer International Publishing, DOI: 10.1007/978-3-319-10064-7_12
    """

    T = float(T)
    n = float(n)
    H = float(H)

    a = eng.fbm1d(H,n,T)
    fbm = np.asarray([])
    for i in range(len(a)):
        fbm = np.concatenate((fbm,np.asarray(a[i])))

    #plt.plot(np.linspace(0,1,int(n+1)),fbm)
    
    return fbm
