import numpy as np
import monti_carlo as mc

import matplotlib.pylab as plt
from scipy.stats import gaussian_kde
import scipy.integrate as itge

def gaussian_random_walk():
  # gaussian random walk
  N = 100

  mean0 = 0.0
  sigma = 3.0

  X0 = mc.monti_carlo_samples(mc.gaussian_random_qfunc,1,(0.0,3.0))

  X = mc.markov_chain_conditionaldistribution(mc.gaussian_random_qfunc,X0,N,(3.0,))

  t = np.arange(N)
  plt.plot(t, X, 'kx-')
  plt.show()
  plt.clf()

if __name__ == '__main__':
  gaussian_random_walk()

