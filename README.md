# mncPy
Memory-friendly parallel Python scripts to manipulate netcdf files

## Prerequisites:

On cheyenne, load the necessary modules by running the following commands before running the scripts:
```
module load python
ncar_pylib
```

Note: The scripts must be run on multiple MPI tasks via interactive jobs or batch jobs. A typical usage:

```
module load python
ncar_pylib
mpiexec -n 36 compress.py -f *.nc
```

To run parallel jobs on cheyenne, see https://www2.cisl.ucar.edu/resources/computational-systems/cheyenne/running-jobs/submitting-jobs-pbs


------------

## compress.py:
```
usage: compress.py [-h] -f [path [path ...]] [-x excl] [-c comp] [-v]

Compresses a given set of netcdf files in parallel

optional arguments:
  -h, --help            show this help message and exit
  -f [path [path ...]]  path to hist file(s) to read. Multiple files and wild
                        characters (*) accepted.
  -x excl               (optional). File names that have the string provided
                        after this flag will be discarded.
  -c comp               (optional). 3-character CESM component name that
                        generates the history files. If not provided, the
                        script will try to determine this from file names.
  -v                    Verbose logging
```
