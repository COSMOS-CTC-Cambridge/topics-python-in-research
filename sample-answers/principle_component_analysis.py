import numpy as np
import scipy.linalg as lalg

 #ar1/2 is a 2D array of vectors
def covarience_matrix(ar1,ar2):
  
  #far1 = np.fft.rfftn(ar1)
  #far2 = np.fft.rfftn(ar2)

  Ncomponents = far1.shape[0]

  #N1 = ar1.shape[0]
  #N2 = ar2.shape[0]

  CovMat = np.zeros((Ncomponents,Ncomponents))

  Nfreq = min(ar1.shape[1],ar2.shape[1])

  far1 = np.fft.rfftn(ar1,axis=1,s=Nfreq)
  far2 = np.fft.rfftn(ar2,axis=1,s=Nfreq)

  for i in xrange(Ncomponents):
    for j in xrange(Ncomponents):
      fcoverience = far1[i]*far2[j]
      CovMat[i,j] = np.sum(np.fft.irfftn(fcoverience,axis=1,s=Nfreq))

  return CovMat


# ok abuse of terminology will be rife here.

def autocorrelation(ar1):
  return covarience_matrix(ar1,ar1)

def PCA(ar1):

  covmat = autocorrelation(ar1)
 
  eigval, eigvec = lalg.eig(ar1)

  retar = np.zeros_like(ar1)

  for i in xrange(eigval.size):  
    retar[i] = np.dot(eigvec[i],ar1)

  return retar



