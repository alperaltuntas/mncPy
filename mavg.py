#!/usr/bin/env python

""" Generates weighed monthly avg files from given history files with more frequently recorded data.
    The monthly files will be saved with a prefix "mavg_".
    After having checked the average files are ok, you can remove the prefix from all files at once
    by running the following linux command (Warning: This will override any existing file, averaged or not!):
        rename "mavg_" "" mavg_*
"""

from __future__ import print_function
import argparse
import netCDF4 as nc4
import xarray as xr
import cftime as cft
from mpi4py import MPI
from common import GlobalData, get_file_paths, AvgInterval, next_month_1st, add_months

# pylint: disable=line-too-long, bad-whitespace, len-as-condition, invalid-name

# Parse the CLI arguments provided by the user
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('-f', metavar='path', type=str, required=True, nargs='*',
                    help='path to hist file(s) to read. Multiple files and wild characters (*) accepted.')
parser.add_argument('-x', metavar='excl', type=str, required=False,
                    help='(optional). File names that have the string provided after this flag'
                         ' will be discarded. ')
parser.add_argument('-d0', metavar='date0', type=str, required=False,
                    help='Beginning date for averaging (yyyy-mm). If not provided, averaging will begin '
                    'from earliest possible date')
parser.add_argument('-d1', metavar='date1', type=str, required=False,
                    help='Ending date for averaging (yyyy-mm). If not provided, averaging will end '
                    'at latest possible date')
parser.add_argument('--compress', action='store_true', required=False, help='If provided, '
                    'write the mavg files in lossless compressed format.')
parser.add_argument('-v', action='store_true', required=False, help='Verbose logging')
args = parser.parse_args()

# MPI comm
comm = MPI.COMM_WORLD

# Global data that should be common to all processors
glob = GlobalData()

def preprocess_out_files():
    """ Extracts some general information about the averaging files to generate. Mainly, determines
        which files to read and weights for each averaging file. Also returns avg_intervals """

    if args.v and comm.Get_rank()==0:
        print("pre-processing files")

    # beginning and ending dates for this proc

    date0_proc = add_months(glob.date0_out, min(glob.m_per_proc*(comm.Get_rank()),glob.nmonths))
    date1_proc = add_months(glob.date0_out, min(glob.m_per_proc*(comm.Get_rank()+1),glob.nmonths))

    # determine the averaging intervals (months) to process for this proc:
    avg_intervals = []
    interval_date0 = date0_proc # first intervals's date0
    while True:
        interval_date1 = next_month_1st(interval_date0) # date1 of current interval
        if interval_date1<=date1_proc:
            avg_intervals.append( AvgInterval(interval_date0,"month",glob.fprefix, glob.fsuffix) )
            interval_date0 = interval_date1 # move to next potential interval
        else:
            # the end of this interval is beyond the extent of input files.
            # do not add this interval and break here.
            break

    fi = 0 # file index
    for interval in avg_intervals:

        interval_preprocessed = False
        while not interval_preprocessed:

            # Read the input file
            with xr.open_dataset(glob.filePaths[fi](),decode_times=False, cache=False, decode_cf=False) as ncfile_in:
                day0_in = ncfile_in[glob.time_bound_str].data[0][0]   # beginning day of this input file
                day1_in = ncfile_in[glob.time_bound_str].data[-1][1]  # ending day of this input file
                date0_in = nc4.num2date(day0_in, glob.nc_dtime_unit, glob.nc_calendar) # beginning date of this file
                date1_in = nc4.num2date(day1_in, glob.nc_dtime_unit, glob.nc_calendar) # ending date of this file

                if (date1_in <= interval.date0):   # The dates of this file (fi) precede the time interval that
                                                # this proc is responsible for. So, move on to next file. (fi++)
                    pass

                elif (date0_in >= interval.date1): # The dates of this file (fi) are later than the time interval that
                                                # this file (fi) covers. so, stop reading files.
                    interval_preprocessed = True

                elif (date0_in <= interval.date0 and date1_in >  interval.date0) or \
                     (date0_in >= interval.date0 and date1_in <= interval.date1) or \
                     (date0_in <  interval.date1 and date1_in >= interval.date1):
                    # The dates of this file (fi) falls within the time interval that this proc is responsible for.
                    # so, read in this file.

                    # Determine the weight of this file (fi):
                    ndays_within_interval = ( min(interval.date1,date1_in)-max(interval.date0,date0_in) ).days
                    weight = float(ndays_within_interval)/interval.ndays
                    interval.in_files.append([glob.filePaths[fi], weight])

                else:
                    print(date0_in, date1_in, interval.date0, interval.date1)
                    raise RuntimeError("Error detected in the file read/write logic", comm.Get_rank())

                # check if this was the final file to read for this interval
                if date1_in>=interval.date1:
                    interval_preprocessed=True

            fi = fi+1 # skip to next file to read, which may have information for this interval (month)

        # Done preprocessing this interval (month). Move to the next interval
        fi = fi-1 # (and move to previous file in case it had information for the next interval)

    return avg_intervals

def process_out_files(avg_intervals):
    """ Generates the average files """

    if args.v and comm.Get_rank()==0:
        print("processing files")

    comm.Barrier()

    compr_dict = dict()
    if args.compress:
        compr_dict = dict(zlib=True, complevel=1)
    compr_dict['_FillValue'] = None
    for interval in avg_intervals:
        if args.v:
            index = avg_intervals.index(interval)
            print("rank:",comm.Get_rank(), " \tprocessing", index+1, "of", len(avg_intervals))

        # instantiate the first input file to get some general information and time-independent arrays:
        in_ds0 = xr.open_dataset(interval.in_files[0][0](), decode_times=False, cache=False, decode_cf=False)

        # instantiate the file to write
        with xr.Dataset(coords=in_ds0.coords, attrs=in_ds0.attrs) as out_ds:

            for da in in_ds0.variables:
                if not glob.time_str in in_ds0[da].dims:
                    out_ds[da] = in_ds0[da]
            encoding_dict = {da: compr_dict for da in in_ds0.variables if not glob.time_str in in_ds0[da].dims}
            out_ds.to_netcdf(path=interval.out_filename, mode='w',unlimited_dims=[glob.time_str], encoding=encoding_dict)

        # now compute and fill in variables with "time" dimension
        # (weighted averaging)
        for da in in_ds0.variables:

            if da in [glob.time_str, glob.time_bound_str]:
                continue

            if glob.time_str in in_ds0[da].dims:
                if  args.v and comm.Get_rank()==0:
                    print(da)

                # re-construct and destruct dataset for each variable to reduce memory hit
                # such that the dataset stores only one variable in primary memory at a time
                with xr.Dataset(coords=in_ds0.coords, attrs=in_ds0.attrs) as out_ds:

                    for in_file in interval.in_files:
                        in_filepath = in_file[0]()
                        weight      = in_file[1]
                        with xr.open_dataset(in_filepath, decode_times=False, cache=False, decode_cf=False) as in_ds:
                            if not da in out_ds: # instantiate the data array
                                out_ds[da]      = in_ds[da]             # first, copy the array with all tha metada
                                out_ds[da].data = in_ds[da].data*weight # now correct the values
                            else: # accumulate
                                out_ds[da].data = out_ds[da].data + in_ds[da].data*weight

                    # Reapply mask
                    if ('_FillValue' in out_ds[da].attrs) and ('_FillValue' in in_ds[da].attrs):
                        fillVal = in_ds[da].attrs['_FillValue']
                        out_ds[da].data = xr.where( in_ds[da].data == fillVal, fillVal, out_ds[da].data)
                    #else:
                    #    print("Cannot find '_FillValue' for "+da)

                    out_ds.to_netcdf(path=interval.out_filename, mode='a', encoding={da:compr_dict})

        # finally, correct time and time_bound
        with xr.open_dataset(in_filepath, decode_times=False, cache=False, decode_cf=False) as in_ds:
            with xr.Dataset() as out_ds:
                # time
                t_str = glob.time_str           # time
                tb_str = glob.time_bound_str    # time_bound
                out_ds[t_str] = in_ds[t_str]    # instantiate time da
                out_ds[tb_str] = in_ds[tb_str]  # instantiate time_bound da

                # update time_bound first
                out_ds[tb_str].data = [[cft.date2num(interval.date0, in_ds[t_str].units, in_ds[t_str].calendar),
                                        cft.date2num(interval.date1, in_ds[t_str].units, in_ds[t_str].calendar) ]]

                # now update time
                out_ds.assign_coords(t_str = [cft.date2num(interval.date1, in_ds[t_str].units, in_ds[t_str].calendar)])
                out_ds[t_str].data[:] = [cft.date2num(interval.date1, in_ds[t_str].units, in_ds[t_str].calendar)]

                # write to file
                out_ds.to_netcdf(path=interval.out_filename, mode='a', encoding={t_str:compr_dict,
                                                                                 tb_str:compr_dict})


def main():
    """ main function"""
    global glob

    # get the list of files
    filePaths = get_file_paths(args.f, comm.Get_rank(), args.x, args.v)

    # determine beginning and ending dates for files to be generated
    user_date0_out = user_date1_out = None
    if args.d0:
        user_date0_out = cft.DatetimeNoLeap(year    = int(args.d0[0:4]),
                                            month   = int(args.d0[5:7]),
                                            day     = 1 )
    if args.d1:
        user_date1_out = next_month_1st(
                            cft.DatetimeNoLeap( year    = int(args.d1[0:4]),
                                                month   = int(args.d1[5:7]),
                                                day     = 1 ) )
    # now get the global information
    glob.obtain_global_info(filePaths, comm, user_date0_out, user_date1_out)

    # broadcast/receive the global information:
    glob = comm.bcast(glob, root=0)

    avg_intervals = preprocess_out_files()

    # print global information
    if comm.Get_rank()==0:
        print("Input begin date: ",glob.date0_in )
        print("Input end date: ", glob.date1_in )
        print("Output begin date: ",glob.date0_out )
        print("Output end date: ",glob.date1_out )
        print("Number of monthly avgs to generate", glob.nmonths)
        print("Max number of months per core:", glob.m_per_proc)

    # generate the avg files:
    process_out_files(avg_intervals)

    if comm.Get_rank()==0:
        print("done.")

if __name__ == "__main__":
    if comm.Get_rank()==0:
        print("running with "+str(comm.Get_size())+" processors...")
    main()
