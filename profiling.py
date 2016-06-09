def afunc(x):
    return x*x

def bfunc(x, y):
    return afunc(x)+y

def cfunc(x, y, z):
    return afunc(x)+bfunc(y,z)

print cfunc(1,2,3)
