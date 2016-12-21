import numpy as np
import matplotlib.pylab as plt
import scipy.signal as sig

 # 1 alive
 # 2 dead
def update_array(ar1):
  convolvemask = np.ones((3,3), dtype=np.int)
  convolvemask[1,1] = 0

  Nlive = sig.convolve2d(ar1,convolvemask, mode='same')

  retar = np.where(ar1==1,
                   np.where(Nlive<2,0,np.where(Nlive>3,0,1)),
                   np.where(Nlive==3,1,0)
                  )

  return retar

  # adds glider to ar1
  # at index loc, the
  # index of the top right
  # cell of the bounding box
def add_glider(ar1,loc):
  glider = np.array([[0,1,0], 
                     [0,0,1],
                     [1,1,1]], dtype=np.int)

  index1 = tuple(loc[0] + np.array([0,1,2]*3))
  index2 = tuple(loc[1] + np.array([0]*3 + [1]*3 + [2]*3))

  ar1[index1,index2] = glider.flatten()
  return ar1
  
  
def add_block(ar1,loc):
  index1 = tuple(loc[0] + np.array([0,1]*2))
  index2 = tuple(loc[1] + np.array([0]*2 + [1]*2))

  ar1[index1,index2] = np.ones(4)
  return ar1

def add_subarray(ar1,subar,loc):
  N = subar.shape[0]
  M = subar.shape[1]

  index1 = tuple(loc[0] + np.array(range(M)*N))
  index2 = tuple(loc[1] + np.outer(range(N),np.ones(M)).flatten())
 
  #print index1
  #print index2

  ar1[index1,index2] = subar.flatten()
  return ar1


  # adds glider gun to ar1
  # at index loc, the 
  # index of the top right
  # cekk if the bounding box
def add_gospergun(ar1,loc):
  glider_gun = np.zeros((10,36), dtype=np.int)

  glider_gun = add_block(glider_gun,(4,0))
  glider_gun = add_block(glider_gun,(2,34))

  component1 = np.array([[0,0,1,1,0,0,0,0],
                         [0,1,0,0,0,1,0,0],
                         [1,0,0,0,0,0,1,0],
                         [1,0,0,0,1,0,1,1],
                         [1,0,0,0,0,0,1,0],
                         [0,1,0,0,0,1,0,0],
                         [0,0,1,1,0,0,0,0]], dtype=np.int)

  component2 = np.array([[0,0,0,0,1],
                         [0,0,1,0,1],
                         [1,1,0,0,0],
                         [1,1,0,0,0],
                         [1,1,0,0,0],
                         [0,0,1,0,1],
                         [0,0,0,0,1]], dtype=np.int)

  glider_gun = add_subarray(glider_gun,component1.transpose(),(2,10))
  glider_gun = add_subarray(glider_gun,component2.transpose(),(0,20))
 
  ar1 = add_subarray(ar1,glider_gun,loc)
  return ar1

def add_smallgrower(ar1,loc):
  grower = np.array([[1,1,1,0,1],
                     [1,0,0,0,0],
                     [0,0,0,1,1],
                     [0,1,1,0,1],
                     [1,0,1,0,1]], dtype=np.int)

  index1 = tuple(loc[0] + np.array(range(5)*5))
  index2 = tuple(loc[1] + np.array([0]*5 + [1]*5 + [2]*5 + [3]*5 + [4]*5))

  ar1[index1,index2] = grower.flatten()
  return ar1

  # simple script to set up
  # and plot results of Game
  # of  Life
if __name__ == '__main__':
  
  N = 100.0

  domain = np.zeros((N,N), dtype=np.int)
  
  domain = add_glider(domain,(10,50))
  domain = add_smallgrower(domain,(70,70))

  domain = add_gospergun(domain,(20,20))

  # note infinate loop
  # exits with ctrl-c
  while True:
    plt.imshow(domain)
    #plt.show()
    plt.pause(.1)
    plt.draw()    
    
    plt.clf()

    domain = update_array(domain)





 

