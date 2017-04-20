# Julius B. Kirkegaard 11/01/17
# jbk28@cam.ac.uk
import numpy as np 
import matplotlib.pyplot as plt

def to_vector(mat):
    return np.ravel(mat)
def to_matrix(vec):
    return np.reshape(vec, shape)

### Define grid
dx = 0.02
x = np.arange(0, 1 + dx, dx)
m = len(x)
X, Y = np.meshgrid(x, x)
shape = X.shape

# Transfer to vectors
x = to_vector(X)
y = to_vector(Y)
n = len(x)

# Laplacian
L = np.zeros((n, n))
for i in range(n):
    L[i,i] = -4
    
    j = np.argmin( (x[i] + dx - x)**2 + (y[i] - y)**2 )
    if i!=j: L[i,j] = 1
    
    j = np.argmin( (x[i] - dx - x)**2 + (y[i] - y)**2 )
    if i!=j: L[i,j] = 1
    
    j = np.argmin( (x[i] - x)**2 + (y[i] + dx - y)**2 )
    if i!=j: L[i,j] = 1
    
    j = np.argmin( (x[i] - x)**2 + (y[i] - dx - y)**2 )
    if i!=j: L[i,j] = 1
L = L/dx**2

# Flow
vx = 20 * (y - 0.5)
vy = -20 * (x - 0.5)

G = np.zeros((n, n))
for i in range(n):
	# x-derivative
	j = np.argmin( (x[i] + dx - x)**2 + (y[i] - y)**2 )
	if i!=j: G[i, j] = vx[i]/(2*dx)
	j = np.argmin( (x[i] - dx - x)**2 + (y[i] - y)**2 )
	if i!=j: G[i, j] = -vx[i]/(2*dx)

	# y-derivative
	j = np.argmin( (x[i] - x)**2 + (y[i] + dx - y)**2 )
	if i!=j: G[i, j] = vy[i]/(2*dx)
	j = np.argmin( (x[i] - x)**2 + (y[i] - dx - y)**2 )
	if i!=j: G[i, j] = -vy[i]/(2*dx)

# Form operator
A = L - G

# Boundary conditions
b = np.zeros(n)
for i in range(n):
    if (x[i]==0 or x[i]==1 or y[i]==0 or y[i]==1):
        A[i, :] = 0
        A[i, i] = 1
    
    if x[i] == 0:
        b[i] = np.exp( -10*(y[i]-0.3)**2 )

# Solve
from scipy.linalg import solve
u = solve(A, b)

# Plot
U = to_matrix(u)
plt.imshow(U, extent=(min(x), max(x), max(y), min(y)))
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Temperature distriubtion of plate')
plt.show()
