import numpy as np
import monti_carlo as mc

import matplotlib.pylab as plt
from scipy.stats import gaussian_kde
import scipy.integrate as itge


# markov chain recurance relation
# scattering problem

# two body integration
def twobody(t, y, mass):
  dy1 = y[2]
  dy2 = y[3]

  r = np.sqrt(y[0]**2 + y[1]**2)

  ddy1 = - mass*y[0]/(r**3)
  ddy2 = - mass*y[1]/(r**3)

  return [dy1,dy2,ddy1,ddy2]


# three body scattering
def scatter_recfunc(t,X_t,U):

  # masses
  ms = 1.0e-3
  ma = 1.0
  mb = 10.0

  reduced_mass = ma*mb/(ma+mb)

  # initial condition
  x0 = X_t[:2]
  v0 = X_t[2:]

  #draw U from a distribution
  # make sure it's +ve
  deltat = np.abs(U[4])

  r = itge.ode(twobody)
  r.set_initial_value(X_t, t).set_f_params(ma + mb)
  r.integrate(r.t+deltat)

  x0 = r.y[:2]
  v0 = r.y[2:]

  va = reduced_mass*v0/ma
  us = U[:2]
  theta = U[2]
  impact_angle = U[3]
  
  u = us - va
  absu = np.sqrt(np.sum(u**2))

  deltav_parrallel = 2*ms*absu*((np.sin(theta/2.0))**2)/(ms + ma)
  deltav_per = -ms*absu*np.sin(theta)/(ms + ma)

  deltav1 = deltav_parrallel*np.cos(theta) - deltav_per*np.sin(theta)
  deltav2 = deltav_parrallel*np.sin(theta) + deltav_per*np.cos(theta)

  va += np.array([deltav1,deltav2])

  v0 = ma*va/reduced_mass

  return np.array([x0[0],x0[1],v0[0],v0[1]])



def radom_scatter_problem():
  N = 1000

  ma = 1.0
  mb = 10.0
  mass = ma + mb

  v0 = np.sqrt(mass/10.0)

  X0 = np.array([10.0,0.0,0.0,v0])

  qfargs=(np.array([v0,v0,np.pi/4.0,0.0,1.0]),np.array([1.0,1.0,1.0,1.0,1.0]))

  X = mc.markov_chain_recurance(scatter_recfunc,mc.gaussian_random_qfunc,X0,N,qfargs,ndim=4)

  plt.plot(X[:,0],X[:,1],'bo-')
  plt.show()
  plt.clf()



if __name__ == '__main__':
  radom_scatter_problem()
