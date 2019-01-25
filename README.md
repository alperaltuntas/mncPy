# mncPy
Memory-friendly parallel Python scripts to manipulate netCDF files.

## Prerequisites:

On NCAR machines, the following commands will load all the necessary libraries and modules:
```
module load python/2.7.14
ncar_pylib
```

On other machines, use conda to install mpi4py, netCDF4, xarray, numpy.

## Running the scripts:

The scripts must be run on multiple MPI tasks (using mpiexec or equivalent) via interactive jobs or batch jobs. A typical usage on an interactive session:

```
module load python/2.7.14
ncar_pylib
mpiexec -n 36 compress.py -f *.nc
```

To run parallel jobs on cheyenne, see https://www2.cisl.ucar.edu/resources/computational-systems/cheyenne/running-jobs/submitting-jobs-pbs


------------

## compress.py:
```
usage: compress.py [-h] -f [path [path ...]] [-x excl] [-v]

Compresses a given set of netcdf files in parallel. The compressed files will
be saved with a prefix "cmpr_". After having checked the compressed files are
ok, you can run the following linux command to OVERRIDE all of the original
files with their compressed versions: rename "cmpr_" "" cmpr_*

optional arguments:
  -h, --help            show this help message and exit
  -f [path [path ...]]  path to hist file(s) to read. Multiple files and wild
                        characters (*) accepted.
  -x excl               (optional). File names that have the string provided
                        after this flag will be discarded.
  -v                    Verbose logging
```

## mavg.py:
```
usage: mavg.py [-h] -f [path [path ...]] [-x excl] [-v]

Generates weighed monthly avg files from given history files with more
frequently recorded data. The monthly files will be saved with a prefix
"mavg_". After having checked the average files are ok, you can remove the
prefix from all files at once by running the following linux command (Warning:
This will override any existing file, averaged or not!): 
rename "mavg_" "" mavg_*

optional arguments:
  -h, --help            show this help message and exit
  -f [path [path ...]]  path to hist file(s) to read. Multiple files and wild
                        characters (*) accepted.
  -x excl               (optional). File names that have the string provided
                        after this flag will be discarded.
  -v                    Verbose logging
```
