# import libraries
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
        
def driver():

  f = lambda x: x**6-x-1
  fp = lambda x: 6*x**5-1
  p0 = 2
  a = 0
  b = 10

  Nmax = 100
  tol = 1.e-13

  (p,pstar,info,it) = newton(f,fp,p0,a,b,tol, Nmax)
  print('the approximate root is', '%16.16e' % pstar)
  print('the error message reads:', '%d' % info)
  print('Number of iterations:', '%d' % it)



def newton(f,fp,p0,tol,Nmax):
  """
  Newton iteration.
  
  Inputs:
    f,fp - function and derivative
    p0   - initial guess for root
    tol  - iteration stops when p_n,p_{n+1} are within tol
    Nmax - max number of iterations
  Returns:
    p     - an array of the iterates
    pstar - the last iterate
    info  - success message
          - 0 if we met tol
          - 1 if we hit Nmax iterations (fail)
     
  """



  fa = f(a)
  fb = f(b);
  c = ((a+b)/2)

  if (a<(c-f(c)/fp(c))<b):
    p = np.zeros(Nmax+1);
    p[0] = p0
    for it in range(Nmax):
      p1 = p0-f(p0)/fp(p0)
      p[it+1] = p1
      if (abs(p1-p0) < tol):
        pstar = p1
        info = 0
        return [p,pstar,info,it]
      p0 = p1
    pstar = p1
    info = 1
    return [p,pstar,info,it]
  if (fa*fb>0):
    ier = 1
    astar = a
    return [astar, ier]

#   verify end points are not a root 
    if (fa == 0):
      astar = a
      ier =0
      return [astar, ier]

    if (fb ==0):
      astar = b
      ier = 0
      return [astar, ier]

    count = 0
    d = 0.5*(a+b)
    while (abs(d-a)> tol):
      fd = f(d)
      if (fd ==0):
        astar = d
        ier = 0
        return [astar, ier]
      if (fa*fd<0):
         b = d
      else: 
        a = d
        fa = fd
      d = 0.5*(a+b)
      count = count +1
#      print('abs(d-a) = ', abs(d-a))
      
    astar = d
    ier = 0
    return [astar, ier]
    
        
driver()
