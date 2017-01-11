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

def gaussian(x,mean,sigma):
  return np.exp(np.sum((x - mean)**2)/(2.0*sigma*sigma))/np.sqrt(2*sigma*sigma*np.pi)

 
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


def metropolis_hastings(pdfunc,proposalq,proposalpdf,X0,N,pdfargs,proposalargs,ndim=1):

  X = np.zeros((N,ndim))
  X[0] = X0

  randval = np.random.rand(N-1,ndim)
  U = np.random.rand(N-1)

  for t in range(N-1):
    y = proposalq(randval[t],X[t],*proposalargs) 

    #print pdfunc(X[t],*pdfargs)
    #print proposalpdf(y,X[t],*proposalargs)
 
    acceptance = (pdfunc(y,*pdfargs)*np.array(proposalpdf(X[t],y,*proposalargs)))  \
                   / (pdfunc(X[t],*pdfargs)*np.array(proposalpdf(y,X[t],*proposalargs)))

    #print pdfunc(X[t],*pdfargs)
    #print proposalpdf(y,X[t],*proposalargs)

    acceptance = min(acceptance,1)

    if U[t] <= acceptance:
      X[t+1] = y
    else:
      X[t+1] = X[t]

  return X


def hit_and_run(pdfunc,proposalq,proposalpdf,X0,N,proposalargs,pdfargs,ndim=1):

  X = np.zeros((N,ndim))
  X[0] = X0

  randval = np.random.rand(N-1,ndim) - 0.5
  U = np.random.rand(N-1)

  for t in range(N-1):
    direction = randval[t]/np.sqrt(np.sum(randval[t]**2))
    
    step = proposalq(direction,X[t],*proposalargs)

    y = X[t] + step*direction

    #print step
    
    acceptance = (pdfunc(y,*pdfargs)*np.array(proposalpdf(abs(step),-np.sign(step),y,*proposalargs)))  \
                      / (pdfunc(X[t],*pdfargs)*np.array(proposalpdf(abs(step),np.sign(step),X[t],*proposalargs)))

    #print acceptance
 
    acceptance = min(acceptance,1)

    if U[t] <= acceptance:
      X[t+1] = y
    else:
      X[t+1] = X[t]

  return X

if __name__ == '__main__':

  import matplotlib.pylab as plt
  from scipy.stats import gaussian_kde
  import scipy.integrate as itge 

  def basic_gaussian_monti_carlo():
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

  def gaussian_random_walk():
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

  def radom_scatter_problem():
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


  def conditional_guassian(x,y,mean,sigma):
    if np.max(x) > 100.0 or np.max(y) > 100.0:
      return 0
    elif np.min(x) < -100.0 or np.min(y) < -100.0:
      return 0
    else:
      return gaussian(x-y,mean,sigma)

  def conditional_gqfunc(x,y,mean,sigma):
    return gaussian_random_qfunc(x,mean,sigma) + y

  def flat(x,*pargs):
    return 1.0

  def MCMC_Metropolis_Hastings():
    N = 10000
    qfargs = (np.array([0.0,0.0]),3.0)
    X0 = np.array([0.01,0.01])
    
    X = metropolis_hastings(flat,conditional_gqfunc,conditional_guassian,X0,N,qfargs,(np.array([0.0]),100.0),ndim=2)

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
      return gaussian(sgn*step,mean,sigma)

  def to_scalor_conditional_gqfunc(x,y,mean,sigma):
    xabs = np.sqrt(np.sum(x**2))
    return gaussian_random_qfunc(xabs,mean,sigma)

  def MCMC_Hit_and_Run():
  
    N = 10000
    qfargs = (np.array([0.0,0.0]),3.0)
    X0 = np.array([0.0,0.0])
  
    X = hit_and_run(flat,to_scalor_conditional_gqfunc,conditional_guassian2,X0,N,qfargs,(np.array([0.0]),0.1),ndim=2)

    xy = np.vstack([X[:,0],X[:,1]])
    z = gaussian_kde(xy)(xy)
 # for i in range(5):
#    plt.plot(i*np.ones_like(X[i*N//5:]),X[i*N//5:],'o-')
    plt.scatter(X[:,0],X[:,1],c=z,s=100, edgecolor='')
    plt.colorbar()
    plt.show()
    plt.clf()    


  #Tests

  #basic_gaussian_monti_carlo()

  #gaussian_random_walk()

  #radom_scatter_problem()

  MCMC_Metropolis_Hastings()

  #MCMC_Hit_and_Run()
  





