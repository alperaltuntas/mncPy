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
import netcdftime as nct

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
                                    # others not tested.

    def obtain_global_info(self, files, commsize):
        """ obtain the global information from the set of netcdf files """

        self.files = files
        self.fprefix = re.split(r'\d+-\d+-\d+', self.files[0])[0]
        self.fsuffix = re.split(r'\d+-\d+-\d+', self.files[0])[1]

        # read the time bounds within all the input netcdf files
        self.date0_in, self.date1_in, self.nc_dtime_unit, self.nc_calendar = read_datetime_info(self.files)

        # determine the time bounds for the monthly netcdf files to be written
        self.date0_out = self.date0_in
        if self.date0_in.day != 1:
            self.date0_out = next_month_1st(self.date0_in)
        self.date1_out = first_of_month(self.date1_in)

        # number of months to produce:
        self.nmonths = (self.date1_out.year - self.date0_out.year )*12 + \
                  (self.date1_out.month - self.date0_out.month)

        # Max number of months per processor
        self.m_per_proc = int(np.ceil(float(self.nmonths)/commsize))


def get_file_names(fpath, proc_rank, args_x, args_v):
    """ Returns a list of nc files at a given directory. If the path is not a directory but a file,
        it checks whether the file is a netcdf file, and if so, returns the directory. """
    filelist = []

    # fpath is a list of multiple files:
    if isinstance(fpath, list):
        filelist = [filename for filename in fpath if (".nc" in filename and not (args_x!=None and args_x in filename))]

    # fpath is a single netcdf file:
    elif os.path.isfile(fpath) and fpath.endswith(".nc"):
        filelist.append(fpath)

    # fpath is a single directory containing one or more netcdf files:
    elif os.path.isdir(fpath):
        for filename in os.listdir(fpath):
            if filename.endswith(".nc") and not (args_x!=None and args_x in filename):
                filelist.append(filename)
        if len(filelist) == 0:
            raise RuntimeError("Couldn't find any .nc file in "+str(fpath)+".")
    else:
        raise RuntimeError("Unknown file type: "+str(fpath)+
                           ". Provide a path to an .nc file or a directory with nc files")

    if len(filelist) == 0:
        raise RuntimeError("Couldn't find any .nc files in the given directory.\n")
    if args_v and proc_rank == 0:
        print("Total number of files to read: "+str(len(filelist)))
    return filelist


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
    return nct._netcdftime.DatetimeNoLeap(date_in.year + mth//12, mth%12 +1, 1 )


def first_of_month(date_in):
    """ returns the first day of the month date_in is in """
    return nct._netcdftime.DatetimeNoLeap(date_in.year, date_in.month, 1)


def add_months(date_in,nmonth):
    """ adds the given number of months to a given date """
    year = date_in.year + (date_in.month+nmonth-1)//12
    mth  = (date_in.month+nmonth-1)%12 + 1
    day  = min(date_in.day, calendar.monthrange(1,mth)[1]) # -> this assumes LEAP YEAR
    return nct._netcdftime.DatetimeNoLeap(year,mth,day)


def read_datetime_info(files):
    """ returns the beginning and ending dates of a given list of files """
    time_bounds = []
    nc_calendar = None
    nc_dtime_unit = None
    for filename in files:
        ncfile = xr.open_dataset(filename,decode_times=False)
        if len(ncfile.time_bound.data)!=1:
            raise RuntimeError("Multiple time samples in a single file not supported yet. "
                               "You can exclude files with multiple time samples using the -x flag, "
                               "e.g., -x nday")
        time_bounds.append([ncfile.time_bound.data[0][0],       # begin time
                            ncfile.time_bound.data[-1][1] ])    # end time
        # determine the datetime unit
        if not nc_dtime_unit:
            nc_dtime_unit = ncfile.time.units
        else:
            assert nc_dtime_unit == ncfile.time.units, "Incompatible time units in files!"
        # determine the calendar (noleap expected)
        if not nc_calendar:
            nc_calendar = ncfile.time.calendar
        else:
            assert nc_calendar == ncfile.time.calendar, "Incompatible calendars in files!"

    # check whether there are any skipped time intervals
    time_bounds.sort(key = lambda x:x[0])
    if len(time_bounds)>1:
        for i in range(len(time_bounds)-1):
            assert time_bounds[i][1] == time_bounds[i+1][0], \
                    "Time bounds in the netcdf files are discontinuous, i.e., there may be missing netcdf files!"

    # begin datetime
    date0 = nc4.num2date(time_bounds[0][0], ncfile.time.units, nc_calendar)
    date1 = nc4.num2date(time_bounds[-1][1], ncfile.time.units, nc_calendar)
    return date0, date1, nc_dtime_unit, nc_calendar


class AvgChunk(object):
    """ Encapsulates some general information about average chunks, e.g., monthly averages, to be generated """
    def __init__(self, date0, freq, fprefix, fsuffix):
        self.date0 = date0
        if freq == "m" or freq == "month":
            self.date1 = next_month_1st(self.date0)
            self.ndays = calendar.monthrange(1,self.date0.month)[1] # assumes no leap year

        # list of input files to read to generate this average
        self.in_files = []

        # determine out file name
        hist_str = "{:04d}-{:02d}-{:02d}".format(self.date1.year,self.date1.month,self.date1.day)
        self.out_filename = "mavg_"+fprefix+hist_str+fsuffix
