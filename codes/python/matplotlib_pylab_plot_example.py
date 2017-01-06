filename="files/matplotlib-parabola.png"
import pylab, numpy
x = numpy.mgrid[-5:5:100j]
pylab.plot(x, x**2, "b-", label=r"$x^2$")
pylab.legend()
# this MUST BE CALLED so that the variable "filename" is set, e.g. by
# specifying header argument :var filename="foobar"
pylab.savefig(filename)
print(filename, end="")
