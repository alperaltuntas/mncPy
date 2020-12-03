""" common.py: Common classes and functions used by mncPy scripts
    altuntas@ucar.edu, 2019
"""

from __future__ import print_function
import os
import re
import calendar
import numpy as np
import netCDF4 as nc4
import xarray as xr
import cftime as cft
from collections import namedtuple
from mpi4py import MPI

# pylint: disable=line-too-long, bad-whitespace, len-as-condition

class GlobalData(object,):
    """ Encapsulates global data to be determined by rank 0 and broadcasted to the rest of the processors.
        Distributed memory ensures data encapsulation among tasks anyways, so this is for clearer coding purposes."""
    # pylint: disable=too-many-instance-attributes

    def __init__(self):
        self.files          = None # list of files
        self.fprefix        = None # netcdf file prefix
        self.fsuffix        = None # netcdf file suffix
        self.date0_in       = None # beginning date of the given netcdf input files
        self.date1_in       = None # ending date of the given netcdf input files
        self.date0_out      = None # beginnging date of the monthly netcdf files to be written
        self.date1_out      = None # ending date of the monthly netcdf files to be written
        self.nmonths        = None # number of monthly files to write
        self.m_per_proc     = None # nmonths per processor
        self.nc_calendar    = None # netcdf calendar type. should be no leap. others not tested
        self.nc_dtime_unit  = None # netcdf datetime type. should be "days since 0000-01-01 00:00:00".
        self.time_str       = None # time variable in the netcdf files
        self.time_bound_str = None # time_bound variable in the netcdf files
        self.filePaths      = None # input file paths provided to be read in to generate averagings

    def get_file_naming_pattern(self, filePaths):

        self.filePaths = filePaths
        f0name = self.filePaths[0].name

        # determine the parts of the file name that comes before (prefix) and after (suffix) the date identifier
        ndashes0 = f0name.split('.')[-2].count('-') # if two dashes: year-month-day; if one: year-month
        if ndashes0==1:
            self.fprefix = re.split(r'\d+-\d+', f0name)[0]
            self.fsuffix = re.split(r'\d+-\d+', f0name)[1]
        elif ndashes0==2:
            self.fprefix = re.split(r'\d+-\d+-\d+', f0name)[0]
            self.fsuffix = re.split(r'\d+-\d+-\d+', f0name)[1]
        else:
            raise RuntimeError("Cannot determine the date pattern in file "+f0name)

        for filepath in self.filePaths:
            filename = filepath.name
            ndashes = filename.count('-')
            ndashes = filename.split('.')[-2].count('-')
            if not ndashes==ndashes0:
                raise RuntimeError("Files have different naming patterns. Make sure all files have the same pattern."+\
                                   " You may use -x flag to exclude certain files.")
            if ndashes0==1:
                fprefix = re.split(r'\d+-\d+', f0name)[0]
                fsuffix = re.split(r'\d+-\d+', f0name)[1]
            elif ndashes0==2:
                fprefix = re.split(r'\d+-\d+-\d+', f0name)[0]
                fsuffix = re.split(r'\d+-\d+-\d+', f0name)[1]

            if (not fprefix==self.fprefix) or (not fsuffix==self.fsuffix):
                raise RuntimeError("Files have different naming patterns. Make sure all files have the same pattern."+\
                                   " You may use -x flag to exclude certain files.")

    # get time_str and time_bound_str name:
    def get_time_var_names(self,filePath,bound_required=True):
        """ determines time_str and time_bound_str of a given netcdf file"""

        ds = xr.open_dataset(filePath,decode_times=False)
        if "time" in ds.coords:
            self.time_str = "time"
        elif len(ds.encoding.get("unlimited_dims", None)) ==1:
            self.time_str = list(ds.encoding.get("unlimited_dims", None))[0]
        else:
            raise RuntimeError("Cannot determine time variable")

        if "time_bound" in ds.variables:
            self.time_bound_str = "time_bound"
        elif "bounds" in ds[self.time_str].attrs:
            self.time_bound_str = ds[self.time_str].attrs["bounds"]
        elif bound_required:
            raise RuntimeError("Cannot determine time bound variable")

    def read_datetime_info(self, filePaths, time_str, time_bound_str, comm):
        """ returns the beginning and ending dates of a given list of files """

        tbounds0_local = np.array([-9999]*len(filePaths))
        tbounds1_local = np.array([-9999]*len(filePaths))

        for i in range(comm.Get_rank(), len(filePaths), comm.Get_size()):
            filename = filePaths[i].name
            ncfile = xr.open_dataset(filePaths[i](),decode_times=False)
            if len(ncfile[time_bound_str].data)!=1:
                raise RuntimeError("Multiple time samples in a single file not supported yet. "
                                   "You can exclude files with multiple time samples using the -x flag, "
                                   "e.g., -x nday")
            tbounds0_local[i] = ncfile[time_bound_str].data[0][0]     # begin time
            tbounds1_local[i] = ncfile[time_bound_str].data[-1][1]    # end time
            # determine the datetime unit
            if not self.nc_dtime_unit:
                self.nc_dtime_unit = ncfile[time_str].units
            else:
                assert self.nc_dtime_unit == ncfile[time_str].units, "Incompatible time units in files!"
            # determine the calendar (noleap expected)
            if not self.nc_calendar:
                self.nc_calendar = ncfile[time_str].calendar
            else:
                assert self.nc_calendar == ncfile[time_str].calendar, "Incompatible calendars in files!"

        # reduce the time_bounds globally in rank 0
        tbounds0_global = np.array(tbounds0_local)
        tbounds1_global = np.array(tbounds1_local)
        comm.Reduce([tbounds0_local, MPI.INT],[tbounds0_global, MPI.INT], op=MPI.MAX, root=0)
        comm.Reduce([tbounds1_local, MPI.INT],[tbounds1_global, MPI.INT], op=MPI.MAX, root=0)

        # now make some sanity checks and determine global begin and end times for input files
        comm.Barrier()
        if comm.Get_rank()==0:
            time_bounds = [(tbounds0_global[i], tbounds1_global[i]) for i in range(len(tbounds0_global))]
            time_bounds.sort(key = lambda x:x[0])
            if len(time_bounds)>1:
                # check whether there are any skipped time intervals
                for i in range(len(time_bounds)-1):
                    assert time_bounds[i][1] == time_bounds[i+1][0], \
                            "Time bounds in the netcdf files are discontinuous, i.e., " \
                            "there may be missing netcdf files! Time bounds:"+str(time_bounds[i][1])+\
                            " "+str(time_bounds[i+1][0])

            # global begin and end times
            self.date0_in = nc4.num2date(time_bounds[0][0], ncfile[time_str].units, self.nc_calendar)
            self.date1_in = nc4.num2date(time_bounds[-1][1], ncfile[time_str].units, self.nc_calendar)

        comm.Barrier()
        self.date0_in = comm.bcast(self.date0_in, root=0)
        self.date1_in = comm.bcast(self.date1_in, root=0)

    def obtain_global_info(self, filePaths, comm, user_date0_out=None, user_date1_out=None):
        """ obtain the global information from the set of netcdf files """

        # first some general info. determined by root and broadcasted
        if comm.Get_rank()==0:
            self.get_file_naming_pattern(filePaths)
            self.get_time_var_names(filePaths[0]())
        # now broadcast the general info the root obtained above
        self.fprefix        = comm.bcast(self.fprefix, root=0)
        self.fsuffix        = comm.bcast(self.fsuffix, root=0)
        self.filePaths      = comm.bcast(self.filePaths, root=0)
        self.time_str       = comm.bcast(self.time_str, root=0)
        self.time_bound_str = comm.bcast(self.time_bound_str, root=0)

        # read the time bounds within all the input netcdf files
        self.read_datetime_info(self.filePaths, self.time_str, self.time_bound_str, comm)

        # determine the time bounds for the monthly netcdf files to be written
        if (user_date0_out or user_date1_out) and self.nc_calendar != "noleap":
            raise RuntimeError("-d0 and -d1 options only supported for noleap calendars for now")

        if (user_date0_out and user_date1_out):
            if (not user_date1_out>user_date0_out):
                raise RuntimeError("User specified output end time (-d1) not greater then begin time (-d0)")

        # date0
        self.date0_out = self.date0_in
        if user_date0_out:
            self.date0_out = max(self.date0_out, user_date0_out)
        if self.date0_in.day != 1:
            self.date0_out = next_month_1st(self.date0_in)

        # date1
        self.date1_out = first_of_month(self.date1_in)
        if user_date1_out:
            self.date1_out = min(self.date1_out, user_date1_out )

        # number of months to produce:
        self.nmonths = (self.date1_out.year - self.date0_out.year )*12 + \
                  (self.date1_out.month - self.date0_out.month)

        # Max number of months per processor
        self.m_per_proc = int(np.ceil(float(self.nmonths)/comm.Get_size()))

FilePath_nt = namedtuple('FilePath', ['base','name'])
class FilePath(FilePath_nt):
    def __call__(self):
        return os.path.join(self.base, self.name)

def get_file_paths(fpaths_in, proc_rank, args_x, args_v):

    """ Returns a list of nc files at a given directory. If the path is not a directory but a file,
        it checks whether the file is a netcdf file, and if so, returns the directory. """
    fpaths_out = []

    if isinstance(fpaths_in, list):

        # fpaths_in is a list of multiple files:
        if len(fpaths_in)>1:
            fpaths_out = [FilePath(os.path.split(filepath)[0], os.path.split(filepath)[1]) \
                          for filepath in fpaths_in if (".nc" in filepath and not (args_x!=None and args_x in filepath))]
        elif len(fpaths_in)==1:

            # fpath is a single netcdf file:
            if fpaths_in[0].endswith(".nc"):
                fpaths_out.append(FilePath(os.path.split(fpaths_in[0])[0], os.path.split(fpaths_in[0])[1]))

            # fpaths_in is a single directory containing one or more netcdf files:
            elif os.path.isdir(fpaths_in[0]):
                for filename in os.listdir(fpaths_in[0]):
                    if filename.endswith(".nc") and not (args_x!=None and args_x in filename):
                        fpaths_out.append(FilePath(fpaths_in[0],filename))
            else:
                raise RuntimeError("Unknown file type: "+str(fpaths_in)+
                                   ". Provide a path to an .nc file or a directory with nc files")

    if len(fpaths_out) == 0:
        raise RuntimeError("Couldn't find any .nc files in the given directory.\n")
    if args_v and proc_rank == 0:
        print("Total number of files to read: "+str(len(fpaths_out)))
    return fpaths_out


def determine_component(files,args_c):
    """ determines the CESM component that generated the history file(s)"""
    comp = None
    if args_c is not None:
        comp = args_c.lower()
    elif '.pop.' in files[0]:
        comp = "pop"
    else:
        raise RuntimeError("Cannot determine CESM component that generated the history files.")

    if comp != "pop":
        raise RuntimeError("This CESM component is not supported yet.")
    return comp


def next_month_1st(date_in):
    """ returns the date of the first day of the next month following date_in """
    mth = date_in.month
    return cft.DatetimeNoLeap(date_in.year + mth//12, mth%12 +1, 1 )


def first_of_month(date_in):
    """ returns the first day of the month date_in is in """
    return cft.DatetimeNoLeap(date_in.year, date_in.month, 1)


def add_months(date_in,nmonth):
    """ adds the given number of months to a given date """
    year = date_in.year + (date_in.month+nmonth-1)//12
    mth  = (date_in.month+nmonth-1)%12 + 1
    day  = min(date_in.day, calendar.monthrange(1,mth)[1]) # -> this assumes LEAP YEAR
    return cft.DatetimeNoLeap(year,mth,day)


class AvgInterval(object):
    """ Encapsulates some general information about average intervals, e.g., monthly averages, to be generated """
    def __init__(self, date0, freq, fprefix, fsuffix):
        self.date0 = date0
        if freq == "m" or freq == "month":
            hist_str = "{:04d}-{:02d}".format(self.date0.year,self.date0.month)
            self.date1 = next_month_1st(self.date0)
            self.ndays = calendar.monthrange(1,self.date0.month)[1] # assumes no leap year
        else:
            raise RuntimeError("Unknown interval type")

        # list of input files to read to generate this average
        self.in_files = []

        # determine out file name
        self.out_filename = "mavg_"+fprefix+hist_str+fsuffix
