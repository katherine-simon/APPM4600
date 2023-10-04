# import libraries
import numpy as np
        
def driver():
  
  f = lambda x: np.cos(x)
  s = np.pi/2
  h = 0.01*2.**(-1*np.arange(0,10))\


  (fprime1) = forward(f,s,h)
  print('forward difference is', fprime1)

  (fprime2) = centered(f,s,h)
  print('centered difference is', fprime2)


def forward(f,s,h):
  fprime1 = (f(s+h) - f(s))/h
  return [fprime1]

def centered(f,s,h):
  fprime2 = (f(s+h) - f(s-h)) / (2*h)
  return [fprime2]
    
        
driver()
