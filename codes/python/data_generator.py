import numpy as np
import monti_carlo
import matplotlib.pylab as plt

# generate some data for pca

N = 10000
qfargs = (np.array([0.0,3.0]),np.array([2.0,4.0]))
X = monti_carlo.monti_carlo_samples(monti_carlo.gaussian_random_qfunc,N,qfargs,ndim=2)

a = 0.3
b = 0.7
c = 3.0
d = 0.1

x = a*X[:,0] - b*X[:,1]
y = b*X[:,0] + a*X[:,1] + c*x 

plt.plot(x,y,'ko',mew=2)
plt.show()

np.savetxt('sample_data_for_pca.txt',np.vstack([x,y]),delimiter=',', header='x,y')




