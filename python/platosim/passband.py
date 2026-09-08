import numpy as np
from scipy.interpolate import interp1d
import os

#import importlib.resources as pkg_resources

class Filter:
    def __init__(self, func, wv_min, wv_max):
        self.func = func
        self.wv_min = wv_min
        self.wv_max = wv_max
    
    def __call__(self, wav: np.ndarray):
        try:
            return self.func(wav)
        except:
            return np.array([self.func(x) for x in wav])

    def __hash__(self):     
        hashArray = np.concatenate([np.array([self.wv_min,self.wv_max]) , self.__call__(np.linspace(self.wv_min , self.wv_max , 5))])
        s = tuple(np.array(hashArray*1000, dtype=np.int16))
        return hash(s)
    
    def __equals__(self, other):
        return self.hashCode() == other.hashCode()

    def __repr__(self): #print function
        return("vmin = " + str(self.wv_min) + " |  vmax = " +  str(self.wv_max))

    

    


def tRed(wav):
    
    def rico(pointA , pointB):
        return ( pointB[1] - pointA[1] ) / ( pointB[0] - pointA[0] )
    def vertDist(pointA, pointB):
        a = rico(pointA, pointB)
        return  - a * pointA[0] + pointA[1]
    
    if wav < 655:
        return 0.0000001
    if wav >= 655 and wav <= 700:
        return 0.70
    
    if wav > 700 and wav <= 800:
        point1 = [700, 0.7]
        point2 = [800, 0.55]
        a = rico(point1,point2) 
        b = vertDist(point1,point2)
        
        return a*wav + b
    
    if wav > 800 and wav <= 950:
        point1 = [800, 0.55]
        point2 = [950, 0.17]
        a = rico(point1,point2) 
        b = vertDist(point1,point2)
        
        return a*wav + b
    
    if wav > 950 and wav <= 1030:
        point1 = [950, 0.17]
        point2 = [1000, 0.08]
        a = rico(point1,point2) 
        b = vertDist(point1,point2)
        
        return a*wav + b
    
    if wav > 1030:
        return 0.0000001
    
        
        

def tBlue(wav):
    if wav > 500 and wav <= 600:
        return 0.0014*(wav-500) + 0.58
    elif wav > 600 and wav <= 700:
        return 0.72
    else:
        return 0.0000001


def tVisual(wav):
    if wav > 350 and wav <= 400:
        return (wav-350)/50
    elif wav > 400 and wav <= 660:
        return 1
    elif wav > 660 and wav < 700:
        return 1 + (660-wav)/40
    else:
        return 0.0000001

#Importing the PLATO Bandpass

import pkg_resources

# Could be any dot-separated package/module name or a "Requirement"
resource_package = __name__
resource_path = '/'.join(('files', 'plato_bandpass.txt'))  # Do not use os.path.join()
path = pkg_resources.resource_stream(resource_package, resource_path)
plato_bandpass = np.loadtxt(path, dtype=float)
plato_bandpass_spline = interp1d(plato_bandpass[:,0], plato_bandpass[:,1])


#Making 4 filters R, B, V and P. P uses the imported PLATO Spline
filterR = Filter(tRed, 655, 1030)
filterB = Filter(tBlue, 500, 700)
filterV = Filter(tVisual, 350, 700)
filterP = Filter(plato_bandpass_spline,500,1000)




def fromVtoP(V:float , Teff:float):
    a = -1.184*10**(-12)
    b = 4.526*10**(-8)
    c = -5.805*10**(-4)
    d = 2.449
    polynomial = a*Teff**3 + b*Teff**2 + c*Teff + d
    return V - polynomial

def fromPtoV(P:float , Teff:float):
    a = -1.184*10**(-12)
    b = 4.526*10**(-8)
    c = -5.805*10**(-4)
    d = 2.449
    polynomial = a*Teff**3 + b*Teff**2 + c*Teff + d
    return P + polynomial




