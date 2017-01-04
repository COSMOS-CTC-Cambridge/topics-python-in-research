import numpy
import h5py
import os
import tempfile
import cProfile
import pstats

def h5py_create(filename, datadict, compression):
    '''Create a new HDF5 file called "filename" and save the values of "datadict" into it using its keys as
    the dataset names; create an attribute called "compression" holding the value of "compression" parameter.'''
    f = h5py.File(filename, mode="w")
    attrvalue = "nothing interesting for now"
    f.attrs.create("top-level-attribute", attrvalue, dtype="S{x}".format(x=len(attrvalue)))
    for name,value in datadict.items():
        ds = f.create_dataset(name, data=value, compression=compression, chunks=True)
        ds.attrs.create("compression", str(compression), dtype="S{x}".format(x=len(str(compression))))
    return

def szip_available():
    '''Try to create a dataset using szip: return True if succeeds, False on ValueError (szip not available)
    and raise on others.'''
    import tempfile
    tempf = tempfile.NamedTemporaryFile(dir=".")
    f = h5py.File(tempf.name,"w")
    try:
        f.create_dataset("foo", shape=(10,10), dtype="f8", compression="szip")
    except ValueError:
        ret = False
    else:
        ret = True
    finally:
        f.close()
    return ret

data=numpy.random.random((1000,1000,100))
tempfiles = [tempfile.NamedTemporaryFile(dir=".") for i in [0,1,2,3]]
cps = [cProfile.Profile() for i in range(len(tempfiles))]
if (szip_available()):
    comp="szip"
else:
    comp="gzip"
runs = [None] + 3*[comp]
for i,r in enumerate(runs):
    if (i==2):
        data[100:900,100:900,30:70]=0.0
    if (i==3):
        data = numpy.ones((1000,1000,100), dtype=numpy.float64)
    cps[i].runcall(h5py_create, tempfiles[i].name, {"array_called_data":data}, r)

print('''Time spent writing hdf5 data and file sizes:
  uncompressed random data:   {uncompt:g}\t{uncomps} 
  {comp} compressed random data:     {compt:g}\t{comps}
  {comp} compressed semirandom data: {semit:g}\t{semis}
  {comp} compressed zeros:           {zerot:g}\t{zeros}'''.format(
      uncomps=os.stat(tempfiles[0].name).st_size,
      comps=os.stat(tempfiles[1].name).st_size,
      semis=os.stat(tempfiles[2].name).st_size,
      zeros=os.stat(tempfiles[3].name).st_size,
      uncompt=pstats.Stats(cps[0]).total_tt,
      compt=pstats.Stats(cps[1]).total_tt,
      semit=pstats.Stats(cps[2]).total_tt,
      zerot=pstats.Stats(cps[3]).total_tt,
      comp=comp
  ))
