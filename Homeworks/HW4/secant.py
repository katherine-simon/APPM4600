# import libraries
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
        
def driver():
  f = lambda x: x**6-x-1
  p0 = 2
  p1 = 1

  Nmax = 100
  tol = 1.e-13

  (p,pstar,info,i) = secant(f,p0, p1, tol, Nmax)
  print('the approximate root is', '%16.16e' % pstar)
  print('the error message reads:', '%d' % info)
  print('Number of iterations:', '%d' % i)
  print(abs(p-pstar)[0:i])

  xk = (abs(p-pstar)[0:i-1])
  xk1 = (abs(p-pstar)[1:i])

  plt.plot(xk,xk1)
  plt.xscale("log")
  plt.yscale("log")
  plt.show()

  print("slope", (xk[5]-xk[1])/(xk1[5]-xk1[1]))

def secant(f,p0,p1,tol,Nmax):
  """
  Secant iteration.
  
  Inputs:
    f    - function
    p0   - first initial guess for root
    p1   - second initial guess for root
    tol  - iteration stops when p_n,p_{n+1} are within tol
    Nmax - max number of iterations
  Returns:
    p     - an array of the iterates
    pstar - the last iterate
    info  - success message
          - 0 if we met tol
          - 1 if we hit Nmax iterations (fail)
     
  """

  p = np.zeros(Nmax)
  p[0] = p0
  if abs(f(p0)) == 0:
    pstar = p0
    info = 0
    return [p,pstar,info,i+1]
  if abs(f(p1)) == 0:
    pstar = p1
    info = 0
    return [p,pstar,info,i+1]

  fp0 = f(p0)
  fp1 = f(p1)

  for i in range(Nmax):
    if abs(fp1-fp0) == 0:
      info = 1
      pstar = p1
      return [p,pstar,info,i+1]
    p2 = p1 - (fp1*(p1-p0)/(fp1-fp0))
    if abs(p2-p1) < tol:
      pstar= p1
      info = 0
      return [p,pstar,info,i+1]
    p0 = p1
    fp0 = fp1
    p1 = p2
    fp1 = f(p2)
    p[i+1] = p0
  pstar = p2
  info = 1
  return [p,pstar,info,i+1]

        
driver()
