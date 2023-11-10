import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from scipy import integrate

def driver():

    f = lambda x: 1/(1+x**2)
      
    a = -5
    b = 5
    n1 = 100
    n2 = int(10**3.5)
    
    
    xint =np.linspace(a,b,n1)
    yint = f(xint)
    h = xint[1]-xint[0]
    
    xint2 = np.linspace(a,b,n2)
    yint2 = f(xint2)
    h2 = xint2[1]-xint2[0]

    
    N =len(xint) -2
    N2 = len(xint2) -2

    tol = 1.e-4


    Tn = trapezoidal(f,a,b,xint2,yint2,N2,h2,tol)
    Sn = simpson(f,a,b,xint2,yint2,N2,h2,tol)
    print('Tn is',Tn)
    print('Sn is',Sn)


    print(integrate.quad(f,a,b,epsrel=1e-4))
    Neval = 1000
    xeval = np.linspace(a,b,N)     


def trapezoidal(f,a,b,xint,yint,N,h,tol):

    Tn =0
    countt = 0
    
    for i in range(N+1):
        if abs(integrate.quad(f,a,b)[0] - Tn) > tol:
            Tn += (h/2)*(yint[i]+yint[i+1])
            countt = countt +1
    return (Tn,countt)


def simpson(f,a,b,xint,yint,N,h,tol):

    Sn = 0
    counts = 0
    for i in range(0,N,2):
        if abs(integrate.quad(f,a,b)[0] - Sn) > tol:
            Sn += (h/3)*(yint[i]+4*yint[i+1]+yint[i+2])
            counts = counts+1
    return (Sn,counts)


    
driver()    
       
