# import libraries
import numpy as np
    
def driver():

# test functions 
     f1 = lambda x: (10/(x+4))**0.5
# fixed point is alpha1 = 1.4987....

#     f2 = lambda x: 3+2*np.sin(x)
#fixed point is alpha2 = 3.09... 

     Nmax = 100
     tol = 1e-10

# test f1 '''
     x0 = 1.5
     [xstar,ier,vec,c] = fixedpt(f1,x0,tol,Nmax)
     [avec,acount] = atikens(vec,tol)
     print('the approximate fixed point is:',xstar)
     print('f1(xstar):',f1(xstar))
     print(vec)
#     print(evec)
     print(c)
     print('Error message reads:',ier)
     print(avec)
     print(acount)
    
#test f2 '''
#     x0 = 0.0
#     [xstar,ier,vec] = fixedpt(f2,x0,tol,Nmax)
#     print('the approximate fixed point is:',xstar)
#     print('f2(xstar):',f2(xstar))
#     print('Error message reads:',ier)



# define routines
def fixedpt(f,x0,tol,Nmax):
     count = 0
     vec = []
#     evec = []
     p = 1.3652300134140976
     c = 0
     while (count <Nmax):
          vec.append(x0)
#          evec.append(x0-p)
          count = count +1
          x1 = f(x0)
          if (abs(x1-x0) <tol):
               xstar = x1
               ier = 0
               c=c+np.abs((x1-p)/(x0-p))
               print(count)
               return [xstar,ier,np.array(vec),c]
          x0 = x1

          xstar = x1
          ier = 1
     print(count)
     return [xstar, ier, np.array(vec),c]

def atikens(vec,tol):
     avec = []
     acount = 0
     
     for i in range(len(vec)):
          num = (vec[i+1]-vec[i])**2
          denom = vec[i+2]-2*vec[i+1]+vec[i]
          avec.append(vec[i]-(num/denom))
          acount = acount +1

          if (abs(avec[i+1]-avec[i])<tol):
               print(acount)
               return[np.array(avec),acount]
     
     print('acount is', acount)
     return[np.array(avec),acount]

driver()
