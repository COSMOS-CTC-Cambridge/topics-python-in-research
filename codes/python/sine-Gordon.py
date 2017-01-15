import numpy
import scipy.optimize
import matplotlib.pyplot

class sineGordon:
    '''A class to represent the discretised energy of the sine-Gordon field with
    given lattice spacing, peak location, and mass and its gradient. We also
    provide an initial guess generator
    '''
    def __init__(self, lattice_spacing=0.1, mass=1.0, peak_location=0.0):
        self.h = lattice_spacing=0.1
        self.m = mass
        self.a = peak_location
    def initial_guess(self, points, amplitude=0.0):
        '''An initial guess with amplitude size random fluctuations around the exact
        solution. The points parameter determines the number of points to
        use. They will be separated by self.h and symmertically around origin; a
        boundary condition is imposed at both ends.
        >>> sG = sineGordon(lattice_spacing=0.1, mass=2.0, peak_location=0.1)
        >>> sG.initial_guess(5)
        array([ 0.        ,  2.36211041,  2.74423296,  3.14159265,  6.28318531])
        '''
        xmax = (points//2-((points+1)%2)/2)*self.h
        xmin = -xmax
        x = numpy.mgrid[xmin:xmax:1j*points]
        state = (4.0 * numpy.arctan(numpy.exp(self.m*(x-self.a))) +
                 amplitude*(numpy.random.random(x.shape)-0.5))
        state[0] = 0.0
        state[-1] = 2*numpy.pi
        self.x = x
        return state
    def energy_density(self, state):
        gradf = numpy.gradient(state,self.h)
        energy_density = self.h*(gradf**2/2 + self.m**2*(1 - numpy.cos(state)))
        return energy_density
    def energy(self, state):
        '''The energy of the sine-Gordon field: h(f'(x)^2/2 + m^2(1 - \cos(f))).
        >>> sG = sineGordon(lattice_spacing=0.1, mass=2.0, peak_location=0.1)
        >>> print("{x:.8f}".format(x=sG.energy(sG.initial_guess(5))))
        105.32743396
        '''
        return self.energy_density(state).sum()
    def gradient(self, state):
        '''The gradient wrt f of the sine-Gordon field energy. We use the Euler-Lagrange
        equations instead of the discrete gradient because it is difficult to
        deal with points near the boundary. The discrete gradient is
        \dfrac{1}{4h}\(-f(x_{i+2}) + 2 f(x_i) + m^2 \sin(f(x_i)) - f(x_{i-2})\)
        so if i is next to the boundary, either i+2 or i-2 will be outside our
        array!
        >>> sG.gradient(sineGordon(lattice_spacing=0.1, mass=2.0, peak_location=0.1).initial_guess(5))
        array([   0.        ,   99.32137587,  -19.4809996 , -137.59257666,    0.        ])
        '''
        gradgrad = numpy.gradient(numpy.gradient(state,self.h),self.h)
        gradient = self.m**2*numpy.sin(state) - gradgrad
        gradient[0] = 0.0
        gradient[-1] = 0.0
        return gradient
    def plot(self, state):
        matplotlib.pyplot.gcf()
        matplotlib.pyplot.clf()
        matplotlib.pyplot.plot(self.x, state)
        return

sG = sineGordon(lattice_spacing=0.1, mass=1.0, peak_location=0.0)
state = sG.initial_guess(300, amplitude=0.001)
sol = scipy.optimize.fmin_powell(sG.energy, state)
sol = scipy.optimize.fmin_cg(sG.energy, state, fprime=sG.gradient)
sol = scipy.optimize.fmin_bfgs(sG.energy, state, fprime=sG.gradient)
sG.plot(sol)
