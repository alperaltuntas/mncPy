#!/usr/bin/env python

""" Compresses a given set of netcdf files in parallel. The compressed files will be saved with a prefix "cmpr_".
    After having checked the compressed files are ok, you can run the following linux command to OVERRIDE all of
    the original files with their compressed versions:
        rename "cmpr_" "" cmpr_*
"""

from __future__ import print_function
import argparse
import xarray as xr
import numpy as np
from mpi4py import MPI
from common import get_file_names, GlobalData

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

def compress_files(files):
    """ compresses a given set of netcdf files """

    if args.v and comm.Get_rank()==0:
        print("compressing files")
    comm.Barrier()

    # determine files to be compressed by each proc
    nprocs = comm.Get_size()
    nfiles = len(files)
    f_per_proc = int(np.ceil(float(nfiles)/nprocs))
    lfiles = []
    for i in range(comm.Get_rank()*f_per_proc, min(nfiles, (comm.Get_rank()+1)*f_per_proc)):
        lfiles.append(files[i])


    # compress the files:
    compr_dict = dict(zlib=True, complevel=1)
    compr_dict['_FillValue'] = None
    for lfile in lfiles:
        if args.v:
            print("rank:",comm.Get_rank(), "\tcompressing", lfile, "(", lfiles.index(lfile)+1, "of", len(lfiles), ")")

        # first, write the the coordinates
        var_list = None
        with xr.open_dataset(lfile, decode_times=False, cache=False, decode_cf=False) as lfile_ds_in:
            with xr.Dataset(coords=lfile_ds_in.coords, attrs=lfile_ds_in.attrs) as lfile_ds_out:
                var_list = lfile_ds_in.variables
                for var in lfile_ds_in.coords:
                    lfile_ds_out[var] = lfile_ds_in[var]

                encoding_dict = {var: compr_dict for var in lfile_ds_in.coords}
                lfile_ds_out.to_netcdf(path="cmpr_"+lfile, mode='w',unlimited_dims=["time"],
                                       encoding=encoding_dict)

        # now, write the remaining data arrays (one by one to eliminate memory limitation)
        for da in var_list:
            with xr.open_dataset(lfile, decode_times=False, cache=False, decode_cf=False) as lfile_ds_in:
                with xr.Dataset(coords=lfile_ds_in.coords, attrs=lfile_ds_in.attrs) as lfile_ds_out:
                    if not da in lfile_ds_in.coords:
                        lfile_ds_out[da] = lfile_ds_in[da]
                        lfile_ds_out.to_netcdf(path="cmpr_"+lfile, mode='a',
                                               encoding={da:compr_dict})


    comm.Barrier()
    if args.v and comm.Get_rank()==0:
        print("done.")

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
        print("Record begin date: ",glob.date0_in )
        print("Record end date: ", glob.date1_in )

    compress_files(files)


if __name__ == "__main__":
    if comm.Get_rank()==0:
        print("running with "+str(comm.Get_size())+" processors...")
    main()
