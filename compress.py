#!/usr/bin/env python

""" Compresses a given set of netcdf files in parallel. The compressed files will be saved with a prefix "cmpr_".
    After having checked the compressed files are ok, you can run the following linux command to OVERRIDE all of
    the original files with their compressed versions:
        rename "cmpr_" "" cmpr_*
"""

from __future__ import print_function
import os
import argparse
import xarray as xr
import numpy as np
from mpi4py import MPI
from common import get_file_paths, GlobalData

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

def compress_files():
    """ compresses a given set of netcdf files """

    # get the list of files to compress
    filePaths = get_file_paths(args.f, comm.Get_rank(), args.x, args.v)

    # by default, exclude files that are already compressed by this script
    filePaths = [filePath for filePath in filePaths if len(filePath.name)>5 and "cmpr_"!=filePath.name[0:5]]

    if comm.Get_rank()==0:
        print("compressing "+str(len(filePaths))+" files...")
    comm.Barrier()

    # determine files to be compressed by each proc
    nprocs = comm.Get_size()
    nfiles = len(filePaths)
    f_per_proc = int(np.ceil(float(nfiles)/nprocs))
    lfiles = []
    for i in range(comm.Get_rank()*f_per_proc, min(nfiles, (comm.Get_rank()+1)*f_per_proc)):
        lfiles.append(filePaths[i])


    # compress the files:
    compr_dict = dict(zlib=True, complevel=1)
    compr_dict['_FillValue'] = None
    for lfile in lfiles:

        path_in = os.path.join(lfile.base,lfile.name)
        path_out = os.path.join(lfile.base,"cmpr_"+lfile.name)

        if args.v:
            print("rank:",comm.Get_rank(), "\tcompressing", lfile.name, "(", lfiles.index(lfile)+1, "of", len(lfiles), ")")

        # first, write the the coordinates
        var_list = None
        with xr.open_dataset(path_in, decode_times=False, cache=False, decode_cf=False) as lfile_ds_in:
            with xr.Dataset(coords=lfile_ds_in.coords, attrs=lfile_ds_in.attrs) as lfile_ds_out:
                var_list = lfile_ds_in.variables
                for var in lfile_ds_in.coords:
                    lfile_ds_out[var] = lfile_ds_in[var]

                encoding_dict = {var: compr_dict for var in lfile_ds_in.coords}
                lfile_ds_out.to_netcdf(path=path_out, mode='w',unlimited_dims=["time"], encoding=encoding_dict)

        # now, write the remaining data arrays (one by one to eliminate memory limitation)
        for da in var_list:
            with xr.open_dataset(path_in, decode_times=False, cache=False, decode_cf=False) as lfile_ds_in:
                with xr.Dataset(coords=lfile_ds_in.coords, attrs=lfile_ds_in.attrs) as lfile_ds_out:
                    if not da in lfile_ds_in.coords:
                        lfile_ds_out[da] = lfile_ds_in[da]
                        lfile_ds_out.to_netcdf(path=path_out, mode='a', encoding={da:compr_dict})

    comm.Barrier()
    if comm.Get_rank()==0:
        print("done.")


if __name__ == "__main__":
    if comm.Get_rank()==0:
        print("running with "+str(comm.Get_size())+" processors...")
    compress_files()
