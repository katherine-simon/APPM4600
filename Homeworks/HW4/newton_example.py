# import libraries
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
        
def driver():
#f = lambda x: (x-2)**3
#fp = lambda x: 3*(x-2)**2
#p0 = 1.2
  alpha = 0.138*10**-6
  t = 60*60*60*24
  
  f = lambda x: (np.exp(x)-(3*x**2))**3
  fp = lambda x: 3*((np.exp(x)-(3*x**2))**2)*(np.exp(x)-6*x)

  p0 = 4
  Nmax = 100
  tol = 1.e-13

  (p,pstar,info,it) = newton(f,fp,p0,tol, Nmax)
  print('the approximate root is', '%16.16e' % pstar)
  print('the error message reads:', '%d' % info)
  print('Number of iterations:', '%d' % it)
  print(abs(p-pstar)[0:it])

  xk = (abs(p-pstar)[0:it-1])
  xk1 = (abs(p-pstar)[1:it])
  plt.plot(xk,xk1)
  plt.xscale("log")
  plt.yscale("log")
  plt.show()

  print("slope", (np.log(xk1[6])-np.log(xk1[1]))/(np.log(xk[6])-np.log(xk[1])))



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
        
driver()
