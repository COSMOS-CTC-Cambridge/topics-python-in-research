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

  





