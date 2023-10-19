import matplotlib.pyplot as plt
import numpy as np
import math
from numpy.linalg import inv 


def driver():
    
    f = lambda x: 1/(1+(10*x)**2)
    a = -1
    b = 1

    h = b-a/nint

    
    ''' create points you want to evaluate at'''
    Neval = 1000
    xeval =  np.linspace(a,b,Neval)
    
    ''' number of intervals'''
    Nint = 2
    
    '''evaluate the linear spline'''
    yeval = eval_lin_spline(xeval,Neval,a,b,f,Nint)
    
    ''' evaluate f at the evaluation points'''
    fex = np.zeros(Neval)
    for j in range(Neval):
      fex[j] = f(xeval[j]) 
      
    
    plt.figure()
    plt.plot(xeval,fex,'ro-')
    plt.plot(xeval,yeval,'bs-')
    plt.legend()
    plt.show 
     
    err = abs(yeval-fex)
    plt.figure()
    plt.plot(xeval,err,'ro-')
    plt.show() 
    
    

def line_eval(x0,fx0,x1,fx1,xeval):
    m = (fx1-fx0) / (x1-x0)
    return m*(xeval-x0)+fx0
    
def  eval_lin_spline(xeval,Neval,a,b,f,Nint):

    '''create the intervals for piecewise approximations'''
    xint = np.linspace(a,b,Nint+1)
   
    '''create vector to store the evaluation of the linear splines'''
    yeval = np.zeros(Neval) 
    


    for jint in range(Nint):
        ind = np.where((xint[jint]<xeval) & (xint[jint+1]>xeval))
        n = len(ind[0])
        '''find indices of xeval in interval (xint(jint),xint(jint+1))'''
        '''let ind denote the indices in the intervals'''
        '''let n denote the length of ind'''
        
        '''temporarily store your info for creating a line in the interval of 
         interest'''
        a1= xint[jint]
        fa1 = f(a1)
        b1 = xint[jint+1]
        fb1 = f(b1)
        
        for kk in range(n):
            yeval[ind[0][kk]] = line_eval(a1,fa1,b1,fb1,xeval[ind[0][kk]])
 
        '''use your line evaluator to evaluate the lines at each of the points in the interval'''
        '''yeval(ind(kk)) = call your line evaluator at xeval(ind(kk)) with the points (a1,fa1) and (b1,fb1)'''
    return yeval



def eval_m(a,b,Nint,f):
    hi = (b-a)/Nint
    xint = np.linspace(a,b,Nint+1)

    c = np.zeros([Nint-1, Nint-1])
    for i in range(Nint-1):
        for j in range(Nint-1):
            if (i==j):
                c[i][j] = 1/3
            elif (abs(i-j)==1):
                c[i][j] = 1/12
    invc = inv(c)

    
    y = np.zeros([Nint-1,1])
    for k in range(1,Nint-1):
        y[k] = (f(xint[k+1])-2*f(xint[k])+y(xint[k-1])) / (2*hi**2)

    mi = invc.dot(y)
    return(mi)

    
#    1/12*m(i-1) + 1/3 m(i) +1/12 m(i+1) = y(i+1) -2y(i) + y(i-1) / 2h(i)^2
    


def eval_cubic(mi, f, Nint):
    hi = (b-a)/Nint
    xint = np.linspace(a,b,Nint+1)

    si = np.zeros([Nint-1,1])

    c = np.zeros([Nint-1,1])
    for i in range(Nint-1):
        c[i] = f(xint[i])/hi - hi*mi[i]/6
    
    d = np.zeros([Nint-1,1])
    for j in range([Nint-1]):
        d[j] = f(xint[j+1])/hi - hi*mi[j+1]/6
    
    



           
if __name__ == '__main__':
      # run the drivers only if this is called from the command line
      driver()               
