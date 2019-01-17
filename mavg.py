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
from mpi4py import MPI
from common import GlobalData, get_file_names, AvgChunk, next_month_1st, add_months

# pylint: disable=line-too-long, bad-whitespace, len-as-condition, invalid-name

# Parse the CLI arguments provided by the user
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('-f', metavar='path', type=str, required=True, nargs='*',
                    help='path to hist file(s) to read. Multiple files and wild characters (*) accepted.')
parser.add_argument('-x', metavar='excl', type=str, required=False,
                    help='(optional). File names that have the string provided after this flag'
                         ' will be discarded. ')
parser.add_argument('-v', action='store_true', required=False, help='Verbose logging')
args = parser.parse_args()

# MPI comm
comm = MPI.COMM_WORLD

# Global data that should be common to all processors
glob = GlobalData()

def preprocess_out_files():
    """ Extracts some general information about the averaging files to generate. Mainly, determines
        which files to read and weights for each averaging file. Also returns avg_chunks """

    if args.v and comm.Get_rank()==0:
        print("pre-processing files")

    # beginning and ending dates for this proc
    date0_out = add_months(glob.date0_out, min(glob.m_per_proc*(comm.Get_rank()),glob.nmonths))
    date1_out = add_months(glob.date0_out, min(glob.m_per_proc*(comm.Get_rank()+1),glob.nmonths))

    # determine the averaging chunks (months) to process for this proc:
    avg_chunks = []
    chunk_date0 = date0_out # first chunk's date0
    while True:
        chunk_date1 = next_month_1st(chunk_date0) # date1 of current chunk
        if chunk_date1<=date1_out:
            avg_chunks.append( AvgChunk(chunk_date0,"month",glob.fprefix, glob.fsuffix) )
            chunk_date0 = chunk_date1 # move to next potential chunk
        else:
            # the end of this chunk is beyond the extent of input files.
            # do not add this chunk and break here.
            break

    fi = 0 # file index
    for chunk in avg_chunks:

        chunk_preprocessed = False
        while not chunk_preprocessed:

            # Read the input file
            with xr.open_dataset(glob.files[fi],decode_times=False, cache=False, decode_cf=False) as ncfile_in:
                day0_in = ncfile_in.time_bound.data[0][0]   # beginning day of this input file
                day1_in = ncfile_in.time_bound.data[-1][1]  # ending day of this input file
                date0_in = nc4.num2date(day0_in, glob.nc_dtime_unit, glob.nc_calendar) # beginning date of this file
                date1_in = nc4.num2date(day1_in, glob.nc_dtime_unit, glob.nc_calendar) # ending date of this file

                if (date1_in <= chunk.date0):   # The dates of this file (fi) precede the time interval that
                                                # this proc is responsible for. So, move on to next file. (fi++)
                    pass

                elif (date0_in >= chunk.date1): # The dates of this file (fi) are later than the time interval that
                                                # this file (fi) covers. so, stop reading files.
                    chunk_preprocessed = True

                elif (date0_in <= chunk.date0 and date1_in >  chunk.date0) or \
                     (date0_in >= chunk.date0 and date1_in <= chunk.date1) or \
                     (date0_in <  chunk.date1 and date1_in >= chunk.date1):
                    # The dates of this file (fi) falls within the time interval that this proc is responsible for.
                    # so, read in this file.

                    # Determine the weight of this file (fi):
                    ndays_within_chunk = ( min(chunk.date1,date1_in)-max(chunk.date0,date0_in) ).days
                    weight = float(ndays_within_chunk)/chunk.ndays
                    chunk.in_files.append([glob.files[fi], weight])

                else:
                    print(date0_in, date1_in, chunk.date0, chunk.date1)
                    raise RuntimeError("Error detected in the file read/write logic", comm.Get_rank())

                # check if this was the final file to read for this chunk
                if date1_in>=chunk.date1:
                    chunk_preprocessed=True

            fi = fi+1 # skip to next file to read, which may have information for this chunk (month)

        # Done preprocessing this chunk (month). Move to the next chunk
        fi = fi-1 # (and move to previous file in case it had information for the next chunk)

    return avg_chunks

def process_out_files(avg_chunks):
    """ Generates the average files """

    if args.v and comm.Get_rank()==0:
        print("processing files")

    comm.Barrier()

    for chunk in avg_chunks:
        if args.v:
            print("rank:",comm.Get_rank(), " \tprocessing", avg_chunks.index(chunk)+1, "of", len(avg_chunks))

        # instantiate the first input file to get some general information and time-independent arrays:
        in_ds0 = xr.open_dataset(chunk.in_files[0][0], decode_times=False, cache=False, decode_cf=False)

        # instantiate the file to write
        with xr.Dataset(coords=in_ds0.coords, attrs=in_ds0.attrs) as out_ds:

            # write the data arrays that have no "time" dimension, and so do not need averaging.
            # also, supposedly, these arrays should be lightweight, so write at once.
            for da in in_ds0.variables:
                if not "time" in in_ds0[da].dims:
                    out_ds[da] = in_ds0[da]
            out_ds.to_netcdf(path=chunk.out_filename, mode='w',unlimited_dims=["time"])

        # now compute and fill in variables with "time" dimension
        # (weighted averaging)
        for da in in_ds0.variables:
            if "time" in in_ds0[da].dims:
                if  args.v and comm.Get_rank()==0:
                    print(da)

                # re-construct and destruct dataset for each variable to reduce memory hit
                # such that the dataset stores only one variable in primary memory at a time
                with xr.Dataset(coords=in_ds0.coords, attrs=in_ds0.attrs) as out_ds:

                    for in_file in chunk.in_files:
                        in_filename = in_file[0]
                        weight      = in_file[1]
                        with xr.open_dataset(in_filename, decode_times=False) as in_ds:
                            if not da in out_ds: # instantiate the data array
                                out_ds[da] = in_ds[da]*weight
                            else: # accumulate
                                out_ds[da].data = out_ds[da].data + in_ds[da].data*weight

                    out_ds.to_netcdf(path=chunk.out_filename, mode='a')


def main():
    """ main function"""
    global glob

    # get the list of files
    files = get_file_names(args.f, comm.Get_rank(), args.x, args.v)

    # get the global information from files (at rank 0 only)
    if comm.Get_rank()==0:
        glob.obtain_global_info(files, comm.Get_size())

    # broadcast/receive the global information:
    glob = comm.bcast(glob, root=0)

    # print global information
    if comm.Get_rank()==0:
        print("Input begin date: ",glob.date0_in )
        print("Input end date: ", glob.date1_in )
        print("Output begin date: ",glob.date0_out )
        print("Output end date: ",glob.date1_out )
        print("Number of monthly avgs to generate", glob.nmonths)
        print("Max number of months per core:", glob.m_per_proc)

    # generate the avg files:
    avg_chunks = preprocess_out_files()
    process_out_files(avg_chunks)


if __name__ == "__main__":
    if comm.Get_rank()==0:
        print("running with "+str(comm.Get_size())+" processors...")
    main()
