import numpy as np
import monti_carlo as mc

import matplotlib.pylab as plt
from scipy.stats import gaussian_kde
import scipy.integrate as itge

def basic_gaussian_monti_carlo():
    # basic gaussian monti carlo

  N = 10000
  qfargs = (np.array([0.0,0.0]),3.0)
  X = mc.monti_carlo_samples(mc.gaussian_random_qfunc,N,qfargs,ndim=2)

  xy = np.vstack([X[:,0],X[:,1]])
  z = gaussian_kde(xy)(xy)

  plt.scatter(X[:,0],X[:,1],c=z,s=100, edgecolor='')
  plt.colorbar()
  plt.show()
  plt.clf()

if __name__ == '__main__':
  basic_gaussian_monti_carlo()
