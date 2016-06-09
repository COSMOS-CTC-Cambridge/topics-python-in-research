import numpy
import cProfile

def afunc(x):
    return x*x

def bfunc(x, y):
    return afunc(x)+y

def cfunc(x, y, z):
    return afunc(x)+bfunc(y,z)

a=numpy.arange(0,1000000,1).reshape(100,100,100)
b=numpy.arange(1000000,2000000,1).reshape(100,100,100)
c=numpy.arange(2000000,3000000,1).reshape(100,100,100)

cp=cProfile.Profile()
cp.runcall(cfunc, a, b, c)
cp.print_stats(sort="time")

