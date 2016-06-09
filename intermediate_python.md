numpy/scipy/cython
==================

numpy basics
------------

-   always use numpy array types, `array` or `matrix` as appropriate
-   let's generate a \(10^3\) random dataset and look at its dimensions

``` {.python}
import numpy
data=numpy.random.random((10,10,10))
print data.shape
```

-   accessing the array is easy: the object knows all the usual mathematical operations; all these operations are pointwise, so you can just

``` {.python}
print data, data + 2*data/data**0.5 - data**2
```

-   this is also reasonably efficient!
-   we can easily access *slices* (subarrays) as well

    -   `array[a:b]` gives you elements `array[a]` to `array[b-1]`
    -   multidimensional arrays have dimensions separated by commas: `array[a:b,c:d,e:f]`
    -   any missing endpoint is treated as the extreme point (inclusive): missing starting point becomes first element and missing endpoint becomes last element (note that this is inclusive)
    -   negative indices count from the end of the array
    -   note that there is no way to define a slice with a negative endpoint index which includes last element of the array
    -   slices can stride `array[a:b:c]` gives every cth point from a (inclusive) to b (exclusive)
    -   or can slice backwards: `array[a:b:-c]` gives every cth point **backwards** from a (inclusive) to b (exclusive)
    -   if any axis of your slice contains no points at all, you get an empty array: if you want to remove a dimension, use a trivial slice of one element

``` {.python}
array=numpy.linspace(1,10,10)
print array, array[:], array[:4], array[4:]
print array[:-4], array[-4:]
print array[1:8:2], array[8:1:-2], array[1:8:-2]
```

-   slices are also numpy arrays, so they know arithmetic

``` {.python}
print data[3:6,:3,-2:]
print data[3:6,:3,-2:]**2
```

-   slice access can be slow or fast, depending on factors like whether numpy copies the data, what the striding pattern is like, how big the data is etc

    -   could even be faster to process the whole data instead of a slice
    -   no general rule, be prepared to experiment if copy-process-copy back is better than striding in place or something else
-   numpy also has a bunch of *ufunc* functions: they do the obvious thing point-wise:

``` {.python}
print numpy.sin(data)
```

-   they are implemented in C using numpy array's buffer interface, so are probably at least 100x faster than the equivalent from `math` (e.g. `math.sin`)
-   numpy also knows of matrices (and tensors!)

``` {.python}
mat=numpy.matrix(numpy.random.random((3,3)))
mat1=numpy.matrix(numpy.random.random((3,4)))
mat2=numpy.matrix(numpy.random.random((4,3)))
colvec=numpy.matrix(numpy.random.random((1,3)))
rowvec=numpy.matrix(numpy.random.random((3,1)))
print mat, mat**2, rowvec, colvec
print colvec*mat1, mat2*colvec.T, mat2*rowvec
print mat1*colvec # this raises ValueError: (3,4) matrix cannot be multipied from the left by (1,3) matrix
```

-   never think or call an array a matrix or vice versa: **they obey different arithmetic**
-   but to a certain extent, arrays are vectors: they can be broadcast to 1-column and 1-row matrices, but do not have the usual transposes (they DO have a transpose, though), arithmetic is array arithmetic etc

``` {.python}
arrayvec=numpy.random.random((3))
print arrayvec, arrayvec.T 
print arrayvec*mat1
print mat2*arrayvec
```

writing efficient numpy code
----------------------------

-   Let's also take a sneak peek into next topic, profiling while we look at how to do numerics using Laplacian as an example
-   Laplacian is obviously related to PDEs, but the arithmetic is very similar to e.g. a discrete low pass filter like \(y_{i} = x_{i-1} + a
     (y_{i} - x_{i-1})\) or `output[i] :` output[i-1] + a \* (input[i] - output[i-1])= with `output[0] ` input[0]=
-   in python we could do

``` {.python}
  import numpy
  import cProfile
  import time as timemod

  def init_data(sizes):
      return numpy.random.random(sizes)

  def Laplacian(data, lapl, d):
      for ii in range(1,data.shape[0]-1):
          for jj in range(1,data.shape[1]-1):
              for kk in range(1,data.shape[2]-1):
                  lapl[ii,jj,kk] = (
                        (data[ii-1,jj,kk] - 2*data[ii,jj,kk] + data[ii+1,jj,kk])/d[0]*d[1]*d[2] +
                        (data[ii,jj-1,kk] - 2*data[ii,jj,kk] + data[ii,jj+1,kk])/d[1]*d[0]*d[2] +
                        (data[ii,jj,kk-1] - 2*data[ii,jj,kk] + data[ii,jj,kk+1])/d[2]*d[0]*d[1])
      return

  def runone(func):
      d=numpy.array([0.1,0.1,0.1])
      data=init_data((100,100,100))
      lapl=numpy.zeros_like(data)
      cp=cProfile.Profile()
      start = timemod.clock()
      cp.runcall(func, data, lapl, d)
      end = timemod.clock()
      print("cProfile gave total time of {time} s and the following profile.".format(time=end-start))
      cp.print_stats()

  L=runone(Laplacian)
```

-   that took a while (9.4 s on my laptop)! Any ideas why?
-   let's try numpy-style without explicit loops
-   numpy converts operatins between sliced or whole numpy arrays into vectorised loops

    -   note that this can deceive you: how much memory does `array_A ` array<sub>B</sub> + array<sub>C</sub>\*array<sub>D</sub>= consume? How many memory accesses does it contain?

``` {.python}
  import numpy
  import cProfile
  import time as timemod

  def init_data(sizes):
      return numpy.random.random(sizes)

  def Laplacian_numpyic(data, lapl, d):
      lapl[1:-1, 1:-1, 1:-1] = (
              (data[0:-2,1:-1,1:-1] - 2*data[1:-1,1:-1,1:-1] + data[2:,1:-1,1:-1])/d[0]*d[1]*d[2] +
              (data[1:-1,0:-2,1:-1] - 2*data[1:-1,1:-1,1:-1] + data[1:-1,2:,1:-1])/d[1]*d[0]*d[2] +
              (data[1:-1,1:-1,0:-2] - 2*data[1:-1,1:-1,1:-1] + data[1:-1,1:-1,2:])/d[2]*d[0]*d[1])
      return

  L=runone(Laplacian_numpyic)
```

-   that took **0.05 s** on the same laptop!
-   conclusion: **never write a for-loop in python**
-   let's see how cython works and improves performance

    -   everything from `%%cython` to the next empty line will be saved to a tepmorary file, turned into a C code using cython and then compiled into a python module which is then imported
    -   when cython runs, it does not see our current namespace (it is a separate process), so we need to import whatever we use
    -   there is also a special `cimport` command, which imports "into C code"
    -   the `@cython` lines are *decorators* which affect how cython treats the following function: we want no bounds checking on our arrays and we want \(1/0\) to produce \(\infty\) instead of python's `ZeroDivisionError`
    -   this is more or less standard cython preamble
    -   notice also the type definitions in the function definition: **always** type **everything** in cython as if you do not, cython treats them as pytohn objects with all the performance penalty that implies

``` {.python}
  %load_ext Cython
```

``` {.python}
  %%cython
  import cython
  import numpy
  cimport numpy
  @cython.boundscheck(False)
  @cython.cdivision(True)
  def Laplacian_cython1(object[double, ndim=3] data, object[double, ndim=3] lapl, object[double, ndim=1] d):
      lapl[1:-1, 1:-1, 1:-1] = (
              (data[0:-2,1:-1,1:-1] - 2*data[1:-1,1:-1,1:-1] + data[2:,1:-1,1:-1])/d[0]*d[1]*d[2] +
              (data[1:-1,0:-2,1:-1] - 2*data[1:-1,1:-1,1:-1] + data[1:-1,2:,1:-1])/d[1]*d[0]*d[2] +
              (data[1:-1,1:-1,0:-2] - 2*data[1:-1,1:-1,1:-1] + data[1:-1,1:-1,2:])/d[2]*d[0]*d[1])
      return
```

``` {.python}
  L=runone(Laplacian_cython1)
```

-   that took 0.05 s --- was cython not supposed to speed things up?
-   unfortunately as much as numpy likes array-operations, cython dislikes them
-   we'll also introduce the right datatypes: the `double` we used above just happens to be the same as an element of the `numpy.ndarray` we passed Laplacian

``` {.python}
  %%cython
  import cython
  import numpy
  cimport numpy
  DTYPE=numpy.float64
  ctypedef numpy.float64_t DTYPE_t
  @cython.boundscheck(False)
  @cython.cdivision(True)
  def Laplacian_cython2(numpy.ndarray[DTYPE_t, ndim=3] data, numpy.ndarray[DTYPE_t, ndim=3] lapl, numpy.ndarray[DTYPE_t, ndim=1] d):
      cdef int xmax = data.shape[0]
      cdef int ymax = data.shape[1]
      cdef int zmax = data.shape[2]
      cdef int ii, jj, kk
      for ii in range(1,xmax-1):
          for jj in range(1,ymax-1):
              for kk in range(1,zmax-1):
                  lapl[ii,jj,kk] = (
                      (data[ii-1,jj,kk] - 2*data[ii,jj,kk] + data[ii+1,jj,kk])/d[0]*d[1]*d[2] +
                      (data[ii,jj-1,kk] - 2*data[ii,jj,kk] + data[ii,jj+1,kk])/d[1]*d[0]*d[2] +
                      (data[ii,jj,kk-1] - 2*data[ii,jj,kk] + data[ii,jj,kk+1])/d[2]*d[0]*d[1])
      return
```

``` {.python}
  L=runone(Laplacian_cython2)
```

-   there we go: **0.014 s** on the laptop
-   we can do still better: the gcc compiler used does not realise that the lattice constants do not change from lattice site to lattice site, so the `/d[0]*d[1]*d[2]` etc could be done just once and then multiplied (never divide if you can avoid it!) into the stencil:

``` {.python}
  %%cython
  import cython
  import numpy
  cimport numpy
  DTYPE=numpy.float64
  ctypedef numpy.float64_t DTYPE_t
  @cython.boundscheck(False)
  @cython.cdivision(True)
  def Laplacian_cython3(numpy.ndarray[DTYPE_t, ndim=3] data, numpy.ndarray[DTYPE_t, ndim=3] lapl, numpy.ndarray[DTYPE_t, ndim=1] d):
      cdef int xmax = data.shape[0]
      cdef int ymax = data.shape[1]
      cdef int zmax = data.shape[2]
      cdef int ii, jj, kk
      cdef double d1d2bd0=1.0/d[0]*d[1]*d[2], d0d2bd1=1.0/d[1]*d[0]*d[2], d0d1bd2=1.0/d[2]*d[0]*d[1]
      for ii in range(1,xmax-1):
          for jj in range(1,ymax-1):
              for kk in range(1,zmax-1):
                  lapl[ii,jj,kk] = (
                      (data[ii-1,jj,kk] - 2*data[ii,jj,kk] + data[ii+1,jj,kk])*d1d2bd0 +
                      (data[ii,jj-1,kk] - 2*data[ii,jj,kk] + data[ii,jj+1,kk])*d0d2bd1 +
                      (data[ii,jj,kk-1] - 2*data[ii,jj,kk] + data[ii,jj,kk+1])*d0d1bd2)
      return
```

``` {.python}
  L=runone(Laplacian_cython3)
```

-   and down to a healthy **0.005 s**
-   speedup compared to original code is now **1900x**
-   even compared to the vectorised pure python, it is **10x**
-   Profiling
-   we already know cProfile, but let's see what it gives in a more complicated example

``` {.python}
more complicated cProfile
```

-   cython's profiling capabilities are also of interes: in earlier examples, we saw just something like `_cython_magic_c63ab7889ce7cc65e5cd8f75df5d29ae.Laplacian_cython2` and that's all we would have seen even if the cython code would have had deeper call hierarchies: cProfile cannot see into cython without cython giving it a hand
-   this hand is =@cython.profile(True):

``` {.python}
  %%cython
  import cython
  import numpy
  cimport numpy
  DTYPE=numpy.float64
  ctypedef numpy.float64_t DTYPE_t
  @cython.boundscheck(False)
  @cython.cdivision(True)
  @cython.profile(True)
  def Laplacian_cython3_profile(numpy.ndarray[DTYPE_t, ndim=3] data, numpy.ndarray[DTYPE_t, ndim=3] lapl, numpy.ndarray[DTYPE_t, ndim=1] d):
      cdef int xmax = data.shape[0]
      cdef int ymax = data.shape[1]
      cdef int zmax = data.shape[2]
      cdef int ii, jj, kk
      cdef double d1d2bd0=1.0/d[0]*d[1]*d[2], d0d2bd1=1.0/d[1]*d[0]*d[2], d0d1bd2=1.0/d[2]*d[0]*d[1]
      for ii in range(1,xmax-1):
          for jj in range(1,ymax-1):
              for kk in range(1,zmax-1):
                  lapl[ii,jj,kk] = (
                      (data[ii-1,jj,kk] - 2*data[ii,jj,kk] + data[ii+1,jj,kk])*d1d2bd0 +
                      (data[ii,jj-1,kk] - 2*data[ii,jj,kk] + data[ii,jj+1,kk])*d0d2bd1 +
                      (data[ii,jj,kk-1] - 2*data[ii,jj,kk] + data[ii,jj,kk+1])*d0d1bd2)
      return
```

``` {.python}
  L=runone(Laplacian_cython3_profile)
```

-   unfortunately, profiling creates overhead so now our code is now a bit slower
-   turn profiling off for production

-   Debugging

pudb
----

-   
-   by far the best python debugger
-   interface not very good (pydb has better) but

    -   the only debugger capable of breakpointing inside a GUI mainloop
-   if you want a good interface, run interactively in ipython

    -   won't do GUI mainloops interactively
    -   hard to go inside modules you `import`
    -   very hard to use with MPI and more than one rank

        -   there is a way: `mpirun -np 1 ipython your_progran.py : -np 7 screen python your_program.py`
        -   or replace `ipython` with `pudb`
        -   but you need to make sure your interactive thing does not cause timeouts or deadlocks on the others

pdb/pydb
--------

-   pdb comes with python but is rather limited
-   pydb is a slighly more useful but still loses to pudb by a fair margin
-   you can get into the stack trace with `ipython --pdb`
-   qtcreator
-   do you want QtQuick or Qt Proper?
-   QtQuick uses javascript!

