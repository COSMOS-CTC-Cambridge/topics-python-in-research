import numpy
import scipy.optimize
import matplotlib.pyplot

class Rosenbrock:
    '''A class to represent the discretised energy of the sine-Gordon field with given lattice spacing, peak
    location, and mass and its gradient. We also provide an initial guess generator'''
    def __init__(self, lattice_spacing=0.1, a=1.0, b=100.0):
        self.h = lattice_spacing=0.1
        self.a = a
        self.b = b
    def initial_guess(self, points):
        '''An initial guess for x and y.'''
        state = numpy.array(points)
        return state
    def energy(self, state):
        '''The value of the Rosenbrock function
        f(x,y) = (a-x)^2 + b(y-x^2)^2

        >>> _Rb = Rosenbrock(a=1.0, b=100.0)
        >>> _Rb.energy((2.0,3.0))
        101.0
        '''
        x,y = state
        a,b = self.a, self.b
        energy = (a-x)**2 + b*(y-x**2)**2
        return energy
    def gradient(self, state):
        '''The gradient wrt x,y of the Rosenbrock function'''
        x,y = state
        a,b = self.a, self.b
        gradx, grady = 4*b*x**3-4*b*y*x+2*x-2*a, -2*b*x**2+2*b*y
        return numpy.array([gradx,grady])
    def plot(self, state):
        matplotlib.pyplot.gcf()
        matplotlib.pyplot.clf()
        x,y = state
        a,b = self.a, self.b
        minimum = (a,a**2)
        YX = numpy.mgrid[minimum[0]-3.0:minimum[0]+1.0:100j,minimum[1]-3.0:minimum[1]+3.0:100j]
        matplotlib.pyplot.contourf(YX[1], YX[0], self.energy(YX), 50)
        matplotlib.pyplot.scatter(x,y, marker="o")
        return

Rb = Rosenbrock(a=1.0, b=100.0)
state = Rb.initial_guess((2.0,3.0))
sol = scipy.optimize.fmin_powell(Rb.energy, state)
sol = scipy.optimize.fmin_cg(Rb.energy, state, fprime=Rb.gradient)
sol = scipy.optimize.fmin_bfgs(Rb.energy, state, fprime=Rb.gradient)
Rb.plot(sol)
