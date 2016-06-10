#cython: profile=True
import numpy
import cProfile
import cython

@cython.profile(True)
def afunc(x):
    return x*x

@cython.profile(True)
def bfunc(x, y):
    return afunc(x)+y

@cython.profile(True)
def cfunc(x, y, z):
    return afunc(x)+bfunc(y,z)

a=numpy.arange(0,1000000,1).reshape(100,100,100)
b=numpy.arange(1000000,2000000,1).reshape(100,100,100)
c=numpy.arange(2000000,3000000,1).reshape(100,100,100)

cp=cProfile.Profile()
cp.runcall(cfunc, a, b, c)
cp.print_stats(sort="time")

