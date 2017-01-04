filename="files/genfromtxt_example_data.txt"
import numpy
data = numpy.genfromtxt("files/genfromtxt_example_data.txt", comments="#",
                        delimiter="\t", skip_header=3)
print(data)
