import numpy as np
import monti_carlo as mc

import matplotlib.pylab as plt
from scipy.stats import gaussian_kde
import scipy.integrate as itge

def conditional_guassian(x,y,mean,sigma):
  if np.max(x) > 100.0 or np.max(y) > 100.0:
    return 0
  elif np.min(x) < -100.0 or np.min(y) < -100.0:
    return 0
  else:
    return mc.gaussian(x-y,mean,sigma)

def conditional_gqfunc(x,y,mean,sigma):
  return mc.gaussian_random_qfunc(x,mean,sigma) + y

def flat(x,*pargs):
  return 1.0

def MCMC_Metropolis_Hastings():
  N = 10000
  qfargs = (np.array([0.0,0.0]),3.0)
  X0 = np.array([0.01,0.01])

  X = mc.metropolis_hastings(flat,conditional_gqfunc,conditional_guassian,X0,N,qfargs,(np.array([0.0]),100.0),ndim=2)

  xy = np.vstack([X[:,0],X[:,1]])
  z = gaussian_kde(xy)(xy)
 # for i in range(5):
#    plt.plot(i*np.ones_like(X[i*N//5:]),X[i*N//5:],'o-')
  plt.scatter(X[:,0],X[:,1],c=z,s=100, edgecolor='')
  plt.colorbar()
  plt.show()
  plt.clf()

def conditional_guassian2(step,sgn,x,mean,sigma):
    #y = x - step
  if np.max(x) > 100.0:
    return 0
  elif np.min(x) < -100.0:
    return 0
  else:
    return mc.gaussian(sgn*step,mean,sigma)

def to_scalor_conditional_gqfunc(x,y,mean,sigma):
  xabs = np.sqrt(np.sum(x**2))
  return mc.gaussian_random_qfunc(xabs,mean,sigma)


def MCMC_Hit_and_Run():

  N = 10000
  qfargs = (np.array([0.0,0.0]),3.0)
  X0 = np.array([0.0,0.0])

  X = mc.hit_and_run(flat,to_scalor_conditional_gqfunc,conditional_guassian2,X0,N,qfargs,(np.array([0.0]),0.1),ndim=2)

  xy = np.vstack([X[:,0],X[:,1]])
  z = gaussian_kde(xy)(xy)
 # for i in range(5):
#    plt.plot(i*np.ones_like(X[i*N//5:]),X[i*N//5:],'o-')
  plt.scatter(X[:,0],X[:,1],c=z,s=100, edgecolor='')
  plt.colorbar()
  plt.show()
  plt.clf()


if __name__ == '__main__':
  MCMC_Metropolis_Hastings()

  MCMC_Hit_and_Run()


