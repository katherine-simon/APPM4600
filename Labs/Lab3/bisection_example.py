# import libraries
import numpy as np

def driver():

# use routines    
    f = lambda x: np.sin(x)
    a = 0.5
    b = 3*np.pi/4

#    Problem 4.1:
#    We obtained a successful root for a and c but not b.
#    This is because our function has roots at x = 0 and 1,
#    and bisection cannot find the x=0 root.
#    Part a and c contain 1 in the interval of inspection,
#    but since b does not, we cannot fnd the root.


#    Problem 4.2:
#    The behavior was not what I was expecting for b and c.i,
#    but it was what were expecting for a and c.ii.
#    a, we expected a root at 1 and got that to 10^-5 accuracy.
#    b, we expected to get a root of 1, but got an error message
#    indicating a failure. This is most likely becaise there was two roots
#    at x =1 , and the function could not find this.
#    Additionally, we were not expecting to get a root of x = 0 for c.i,
#    but not think the problem may lie in the squared rather than th x=0 value
#    c.ii was what we expected becuase our root wasnot within the interval.

#    f = lambda x: np.sin(x)
#    a = 0.1
#    b = np.pi+0.1

    tol = 1e-7

    [astar,ier] = bisection(f,a,b,tol)
    print('the approximate root is',astar)
    print('the error message reads:',ier)
    print('f(astar) =', f(astar))




# define routines
def bisection(f,a,b,tol):
    
#    Inputs:
#     f,a,b       - function and endpoints of initial interval
#      tol  - bisection stops when interval length < tol

#    Returns:
#      astar - approximation of root
#      ier   - error message
#            - ier = 1 => Failed
#            - ier = 0 == success

#     first verify there is a root we can find in the interval 

    fa = f(a)
    fb = f(b);
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

