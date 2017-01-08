import numpy as np
import scipy.special as spc

 # pdfunc is probability 
 # density function for
 # the sample properties
def monti_carlo_samples(qfunc,N,qfargs,ndim=1):
  randval = np.random.rand(N,ndim)
  return qfunc(randval,*qfargs)

def gaussian_random_qfunc(randval,mean,sigma):
  return mean + sigma*np.sqrt(2)*spc.erfinv(randval*2.0 - 1)

 
# method from xkcd 221
def get_random_number():
  # chosen by fair dice roll
  # guaranteed to be random
  return 1 
 
 
  # produces X_t+1 from reccurance
  # relation given by
  # X_t+1 = recfunc(t,X_t,U_t)
  # where U_t is a random variable
  # selected from the qfunc quantile
def markov_chain_recurance(recfunc,qfunc,X0,N,qfargs,ndim=1):

  X = np.zeros((N,ndim))
  X[0] = X0

  U =  monti_carlo_samples(qfunc,N-1,qfargs)

  for t in range(N-1):
    X[t+1] = recfunc(t,X[t],U[t])

  return X


  # expect qfunc to be of form
  # X_t+1 = qfunc(randint, X_t, qfargs)
def markov_chain_conditionaldistribution(qfunc,X0,N,qfargs,ndim=1):
  
  X = np.zeros((N,ndim))
  X[0] = X0

  randval = np.random.rand(N-1)

  for t in range(N-1):
    X[t+1] = qfunc(randval[t],X[t],*qfargs)

  return X


if __name__ == '__main__':

  import matplotlib.pylab as plt
  from scipy.stats import gaussian_kde  

  # basic gaussian monti carlo

  N = 10000
  qfargs = (np.array([0.0,0.0]),3.0)
  X = monti_carlo_samples(gaussian_random_qfunc,N,qfargs,ndim=2)

  xy = np.vstack([X[:,0],X[:,1]])
  z = gaussian_kde(xy)(xy)

  plt.scatter(X[:,0],X[:,1],c=z,s=100, edgecolor='')
  plt.colorbar()
  plt.show()
  plt.clf()

  # gaussian random walk
  N = 100

  mean0 = 0.0
  sigma = 3.0

  X0 = monti_carlo_samples(gaussian_random_qfunc,1,(0.0,3.0))

  X = markov_chain_conditionaldistribution(gaussian_random_qfunc,X0,N,(3.0,))

  t = np.arange(N)
  plt.plot(t, X, 'kx-')
  plt.show()
  plt.clf()


  # markov chain recurance relation
  # scattering problem

  import scipy.integrate as itge
  
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

    deltat = 1

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

     

  N = 1000

  ma = 1.0
  mb = 10.0
  mass = ma + mb

  v0 = np.sqrt(mass/10.0)

  X0 = np.array([10.0,0.0,0.0,v0])
  


  qfargs=(np.array([v0,v0,np.pi/4.0,0.0]),np.array([1.0,1.0,1.0,1.0]))

  X = markov_chain_recurance(scatter_recfunc,gaussian_random_qfunc,X0,N,qfargs,ndim=4)

  plt.plot(X[:,0],X[:,1],'bo-')
  plt.show()
  plt.clf()

