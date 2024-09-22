#!/home/alvaro/.conda/envs/conda/bin/python
# -*- coding: utf-8 -*-
#
# ~/projects/piss/src/piss_lib.py
#
# info  : common functions for piss project.
# usage : ---
# author: @alvaroggc


# standard libraries
import locale
import warnings
from functools import partial
from glob import glob


# 3rd party packages
import pyproj
import numpy as np
import xarray as xr
import multiprocess as mp
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# function aliases
print = partial(print, flush=True)

# set non-gui
mpl.use('Agg')

# figures configuration
plt.rcParams['figure.dpi'     ] = 200
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8

# general configuration
warnings.filterwarnings('ignore')               # supress deprecation warnings
locale.setlocale(locale.LC_ALL, 'es_CL.UTF-8')  # apply spanish locale settings

# local source
from piss_params import *

# <things> that should be imported from this module
__all__ = ['trans',
           'transxy',
           'proj',
           'ref_datetime',
           'norm',
           'log_message',
           'load_simulation_variable',
           'convert_to_lambert',
           'distance_between_points',
           'convert_point_to_latlon',
           'convert_point_to_xy',
           'xarray_time_iterator']


############################
##### GLOBAL FUNCTIONS #####
############################


def log_message(msg):
    '''
    info: print log header message as "\n:: {string}: ---" and return indent parameter.
    parameters:
        msg : str -> header message to log.
    returns:
        indent : str -> string with spaces of length (len(string) + 5).
    '''

    # logging message
    print(f'\n:: {msg}: ---')

    # indent parameter
    indent = " " * (len(msg) + 5)

    # output
    return indent


def load_simulation_variable(dirin, simid, key='PSL', cyclic=False):
    '''
    info:
    parameters:
    returns:
    '''

    # check files to load
    if (key in ['PSL']):

        # list of netcdf files to load
        fin = sorted(glob(f'{dirin}/data/piss/yData/h1/{simid}/{simid}.*.nc'))

    elif (key in ['PHIS']):

        # list of netcdf files to load
        fin = sorted(glob(f'{dirin}/data/piss/phis/{simid}.*.nc'))

    # logging message
    indent = log_message(f'loading {key}')
    for f in fin: print(f"{indent}{f.split('/')[-1]}")

    # load all files inside dataset container
    da = xr.open_mfdataset(fin,
                           compat='override',
                           coords='minimal')[key]

    # round spatial coordinates
    da['lat'] = da['lat'].astype(np.float32)
    da['lon'] = da['lon'].astype(np.float32)

    # make longitude cyclic
    if cyclic:

        # clone first value to last position
        dalast        = da.sel({'lon': 0}).copy()
        dalast['lon'] = 360
        da            = xr.concat((da, dalast), 'lon')

    # output
    return da


def convert_to_lambert(da, res=90):
    '''
    info:
    parameters:
    returns:
    '''

    def regrid_timestep(da_k):
        '''
        info:
        parameters:
        returns:
        '''

        # project data
        da_k_xy = griddata(points=(X.flatten(), Y.flatten()),
                           values=da_k.data.flatten(),
                           xi=(XN, YN),
                           method='cubic')

        # output
        return da_k_xy

    # create lat-lon grids
    LON, LAT = np.meshgrid(da['lon'].data, da['lat'].data)

    # create transformer
    latlon_to_lambert = pyproj.Transformer.from_crs(latlon_crs, laea_crs)

    # transform curvilinear grid (lat-lon) to cartesian grid (x-y)
    X, Y = latlon_to_lambert.transform(LAT, LON)

    # define new regular x-y grid
    lim    = np.floor(np.abs(X.max() / 100)) * 100
    xn     = np.arange(-lim, lim+1, res)
    yn     = np.arange(-lim, lim+1, res)
    XN, YN = np.meshgrid(xn, yn, indexing='xy')

    # create container of projected data
    newarr = np.zeros((len(da['time']), len(xn), len(yn)))

    # create arguments iterator
    args = xarray_time_iterator(da)

    # create thread pool
    with mp.Pool(processes=25) as pool:

        # compute processing
        results = pool.imap(regrid_timestep, args)

        # process each timestep
        for k, da_k_xy in enumerate(results):

            # add results to container
            newarr[k, ::] = da_k_xy

    # make new container for regular x-y data
    nda = xr.DataArray(data=newarr,
                       dims=['time', 'y', 'x'],
                       coords={'time': da['time'], 'x': xn, 'y': yn},
                       attrs=da.attrs)

    # assign x-y grid coordinates to new data
    nda = nda.assign_coords({'X': (('y', 'x'), XN),
                             'Y': (('y', 'x'), YN)})

    # output
    return nda


def distance_between_points(p1, p2):
    '''
    info:
    parameters:
    returns:
    '''

    # separate coordinates
    x1 = p1[1]
    x2 = p2[1]
    y1 = p1[0]
    y2 = p2[0]

    # convert distance to km
    dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)

    # output (minimum distance)
    return dist


def convert_point_to_latlon(point):
    '''
    info:
    parameters:
    returns:
    '''

    # create transformer
    lambert_to_latlon = pyproj.Transformer.from_crs(laea_crs, latlon_crs)

    # separate coordinates and value
    x = point[1]
    y = point[0]

    # project to lat-lon
    lat, lon = lambert_to_latlon.transform(x, y)

    # transform lon range (from [-180, 180] to [0, 360])
    lon = ((lon + 360) % 360)

    # output
    return [lat, lon]


def convert_point_to_xy(point):
    '''
    info:
    parameters:
    returns:
    '''


    # create transformer
    latlon_to_lambert = pyproj.Transformer.from_crs(latlon_crs, laea_crs)

    # separate coordinates and value
    lon = point[1]
    lat = point[0]

    # transform point to cartesian grid (x-y)
    x, y = latlon_to_lambert.transform(lat, lon)

    # output
    return [y, x]


def xarray_time_iterator(da):
    '''
    info:
    parameters:
    returns:
    '''

    # create iterator
    for t in da['time']:

        # generate iterator item
        yield da.sel({'time': t})




