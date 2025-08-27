#!/home/cr2/agomez/.conda/envs/conda/bin/python
# -*- coding: utf-8 -*-
#
# ~/projects/piss/src/piss_pproc_sf.py
#
# info  : derivate stream function from CESM2 simulations
# usage : ./piss_pproc_sfvp.py <sim_id>
# author: @alvaroggc


# standard packages
import calendar
import datetime as dt
import locale
import os
import re
import subprocess
import sys
import warnings

from functools import partial
from glob import glob
from os.path import (isdir, isfile)

# 3rd party packages
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import cf
import cmocean
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import xarray as xr
from dateutil.relativedelta import relativedelta as rdelta
from mpl_toolkits.basemap import cm
from scipy import (linalg, signal)
from scipy.stats import linregress

# local source
from _piss_lib import *

# supress deprecation warnings
warnings.filterwarnings('ignore')

# apply spanish locale settings
locale.setlocale(locale.LC_ALL, 'es_CL.UTF-8')

# configure matplotlib
plt.rcParams['figure.dpi'] = 300
plt.rcParams['xtick.labelsize'] = 8
# plt.rcParams['ytick.labelsize'] = 8


###########################
##### LOCAL FUNCTIONS #####
###########################


def save_dataset(ds, path, compress=False):
    '''

    info:
    parameters:
    returns:

    '''

    def code_time(unit):
        '''

        info:
        parameters:

        '''

        # reference time and calendar
        tref     = '0001-01-01 00:00:00'
        calendar = 'noleap'

        # normalization parameter
        norm = dt.timedelta(**{unit: 1})

        # normalize time coordinate (days)
        tnorm = (ds['time'].data - cf.dt(tref, calendar=calendar)) / norm

        # update time coordinate values
        ds['time'] = (tnorm).astype(int)

        # update attributes of time coordinate
        ds['time'].attrs['long_name'] = 'time'
        ds['time'].attrs['units']     = f'{unit} since {tref}'
        ds['time'].attrs['calendar']  = calendar

    # create copy of original dataset
    ds = ds.copy()

    # check if data needs to be compressed
    if compress:

        # encoding options
        encoding = {key: {'zlib': True, 'complevel': 5} for key in ds.keys()}

    else:

        # leave encoding parameter as default
        encoding = None

    # code time
    code_time('days')

    # save dataset
    ds.to_netcdf(path, encoding=encoding)

    # output
    return None


def get_variables(args: list[str]):
    '''

    Retrieve variables from input arguments.

    Parameters
    ----------
        args : list[str] -> List of arguments given to script.

    Returns
    -------
        simid : str -> Simulation ID (default is 'lgm_100').

    '''

    # get copy of arguments
    args = args[1:].copy()

    # formats
    simid_fmt = re.compile(r'^lgm_\d{3}?')      # simulation id
    # ra_fmt    = re.compile(r'^ra_(6|24)h$')

    # retrieve variables
    simid = [arg for arg in args if simid_fmt.match(str(arg))]   # sim. ID
    # simid = [arg for arg in args if ra_fmt.match(str(arg))]

    # check arguments
    simid = 'lgm_100' if not simid else simid[0]

    # output
    return (simid)


############################
##### LOCAL PARAMETERS #####
############################


# get variables from input arguments
simid = get_variables(sys.argv)

# directories
homedir = os.path.expanduser('~')               # home directory
projdir = f'{homedir}/projects/piss'
svrdir  = '/mnt/arcus/results/agomez/projects/piss'

dirdata = f'{svrdir}/data/sfvp'  # processed data

# input filename
fin = f'{simid}.h1.sfvp.nc'

# extra parameters
tab = (" " * 4)


###################
##### RUNNING #####
###################


# initial running time
runtime_ini = dt.datetime.now()

# logging message (tab has 4 spaces)
print('\n:: loading data')
print(f'{tab}{fin}')

# load data
ds = xr.open_mfdataset(f'{dirdata}/{fin}', use_cftime=True)

# retrieve years
yrs = np.unique(ds['time.year'].data).astype(int)

# logging message (tab has 4 spaces)
print('\n:: separating dataset to -one file per year-')

# process each year
for yr in yrs:

    # temporal indexer
    idx = {'time': f'{yr:04d}'}

    # extract portion of dataset
    ds_yr = ds.sel(idx)

    # output file
    fout = f'{simid}.sfvp.{yr:04d}.nc'

    # logging message
    print(f'{tab}{fout}')

    # save dataset
    save_dataset(ds_yr, f'{dirdata}/{fout}', compress=True)

# final running time
runtime_end = dt.datetime.now()

# print runtime
runtime = ((runtime_end - runtime_ini) / dt.timedelta(minutes=1))
print(f'\n\n:: runtime: {runtime:.2f} min')

# final loggin message
print('\n:: uwu\n')


























