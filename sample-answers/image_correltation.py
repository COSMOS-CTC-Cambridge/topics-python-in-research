import numpy as np


 # compute correlation
 # by convolution
def correlation(ar1,ar2,s=None):

  if s==None:
    s=ar1.shape

  # must take into account shorter
  # array size due to hessian representation
  # of fourier transform
  fs = tuple([el for el in s] + [s[-1]//2 + 1])

  far1 = np.fft.rfftn(ar1,s=fs)
  far2 = np.fft.rfftn(ar2,s=fs)

  fcovar = far1*far2.convolve().T

  return np.fft.irfftn(fcovar,s=s)


if __name__ == '__main__':

  import sys
  import matplotlib.pylab as plt 

  file1 = sys.argv[1]
  file2 = sys.argv[2]

  # loads in arrays stored as npy
  ar1 = np.load(file1)
  ar2 = np.load(file2)

  correl = correlation(ar1,ar2)

  plt.imshow(correl)
  plt.show()

