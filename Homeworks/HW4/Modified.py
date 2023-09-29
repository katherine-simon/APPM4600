# import libraries
import numpy as np
        
def driver():

  f = lambda x: (np.exp(x)-(3*x**2))**3
  fp = lambda x: 3*((np.exp(x)-(3*x**2))**2)*(np.exp(x)-6*x)
  fpp = lambda x: 9*(3*np.exp(x)*x**4-90*x**4+26*np.exp(x)*x**2-2*np.exp(2*x)*x**2-8*np.exp(2*x)*x+np.exp(3*x)-2*np.exp(2*x))

  f2 = lambda x: f(x) / fp(x)
  f22 = lambda x: 1- (f(x) * fpp(x) / fp(x)**2)
  f3 = lambda x: x - 2*f2(x)
    

  p0 = 3.5

  Nmax = 100
  tol = 1.e-13

  (p,pstar,info,i) = newton(f3,f22,p0,tol, Nmax)
  print('the approximate root is', '%16.16e' % pstar)
  print('the error message reads:', '%d' % info)
  print('Number of iterations:', '%d' % i)
  print((p[1:i]-pstar)/(p[0:i-1] - pstar))


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
  p = np.zeros(Nmax+1);
  p[0] = p0
  for i in range(Nmax):
      p1 = p0-f(p0)/fp(p0)
      p[i+1] = p1
      if (abs(p1-p0) < tol):
          pstar = p1
          info = 0
          return [p,pstar,info,i]
      p0 = p1
  pstar = p1
  info = 1
  return [p,pstar,info,i]
        
driver()
