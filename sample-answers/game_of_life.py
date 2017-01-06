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
  ar1[loc[0]:loc[0]+glider.shape[0],loc[1]:loc[1]+glider.shape[1]] = glider
  return ar1
  
  
def add_block(ar1,loc):
  ar1[loc[0]:loc[0]+2,loc[1]:loc[1]+2] = np.ones((2,2))
  return ar1

def add_subarray(ar1,subar,loc):
  ar1[loc[0]:loc[0]+subar.shape[0],loc[1]:loc[1]+subar.shape[1]] = subar
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

  glider_gun = add_subarray(glider_gun,component1,(2,10))
  glider_gun = add_subarray(glider_gun,component2,(0,20))

  ar1 = add_subarray(ar1,glider_gun,loc)
  return ar1

def add_smallgrower(ar1,loc):
  grower = np.array([[1,1,1,0,1],
                     [1,0,0,0,0],
                     [0,0,0,1,1],
                     [0,1,1,0,1],
                     [1,0,1,0,1]], dtype=np.int)
  ar1[loc[0]:loc[0]+grower.shape[0],loc[1]:loc[1]+grower.shape[1]] = grower
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





 

