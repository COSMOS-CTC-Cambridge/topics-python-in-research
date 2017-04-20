import numpy as np
import scipy.linalg as lalg

 #ar1/2 is a 2D array of vectors
def correllation(ar1,ar2):

  Ncomponents = ar1.shape[0]

  CovMat = np.zeros((Ncomponents,Ncomponents))

  Nfreq = min(ar1.shape[1],ar2.shape[1])

  for i in range(Ncomponents):
    for j in range(Ncomponents):
      CovMat[i,j] = np.dot(ar1[i],ar1[j])

  return CovMat

def correllation_fft(ar1,ar2):
  
  far1 = np.fft.rfftn(ar1,axes=(0,))
  far2 = np.fft.rfftn(ar2,axes=(0,))

  fcorrel  = np.dot(far1,far2.conjugate().T)

  #print fcorrel

  return np.fft.irfftn(fcorrel,(ar1.shape[0],),axes=(0,))

def covarience_matrix(ar1):
  #nar1 = ar1 - np.outer(np.mean(ar1,axis=1),np.ones(ar1.shape[1]))

  N = ar1.shape[1]
  return correllation(ar1,ar1)/(N - 1)
  #return correllation_fft(ar1,ar1)/(N - 1)

def PCA(ar1):
  

  #print ar1.shape

  nar1 = ar1 - np.outer(np.mean(ar1,axis=1),np.ones(ar1.shape[1]))

  covmat = covarience_matrix(nar1)
 
  #print covmat

  eigval, eigvec = lalg.eig(covmat)

  retar = np.zeros_like(ar1)

  for i in range(ar1.shape[0]): 
    retar[i] = np.dot(eigvec[:,i],nar1)

  return retar, eigvec, eigval



if __name__ == '__main__':
  
  import sys
  import matplotlib.pylab as plt

  filename = sys.argv[1]

  data = np.loadtxt(filename,delimiter=',', skiprows=1)   

  pca_data, eigvec, eigval = PCA(data) 
 
  plt.plot(data[0],data[1], 'kx', label='initial')
  plt.plot(pca_data[0],pca_data[1], 'bo', label='pca')
  plt.legend(loc='best')
  plt.show()








