#!/home/alvaro/.conda/envs/conda/bin/python
# -*- coding: utf-8 -*-
#
# ~/projects/piss/src/_piss_lib.py
#
# info  : common functions for piss project.
# usage : ---
# author: @alvaroggc


# standard libraries
import pickle
from functools import partial
from glob import glob


# 3rd party packages
import matplotlib as mpl
import matplotlib.pyplot as plt
import multiprocess as mp
import numpy as np
import pyproj
import xarray as xr
from scipy.interpolate import (griddata, interp1d)

# local source
from _piss_params import *

# <things> that should be imported from this module
__all__ = ['print',
           'latlon_crs',
           'laea_crs',
           'trans',
           'transxy',
           'proj',
           'ref_datetime',
           'norm',
           'log_message',
           'load_simulation_variable',
           'load_results',
           'convert_to_lambert',
           'convert_to_lonlat',
           'distance_between_points',
           'convert_point_to_latlon',
           'convert_point_to_xy',
           'xarray_time_iterator',
           'add_trackpoints',
           'add_trackpoints2']


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
    if ('ra' in simid) and (key != 'PHIS'):

        # list of netcdf files to load (HARDCODED)
        fin = sorted(glob(f'/mnt/fluctus/results/agomez/data/reanalysis/era5/msl_ERA5_SH_*.nc'))

        # change name of variable (differente in reanalysis)
        key = 'msl' if (key == 'PSL') else key

    elif (key in ['PSL']):

        # list of netcdf files to load
        fin = sorted(glob(f'{dirin}/data/piss/yData/h1/{simid}/{simid}.*.nc'))

    elif (key in ['PHIS']):

        # check if reanalysis is needed
        simid = 'lgm_100' if ('ra' in simid) else simid

        # list of netcdf files to load
        fin = sorted(glob(f'{dirin}/data/piss/phis/{simid}.*.nc'))

    # logging message
    indent = log_message(f'loading {key}')
    for f in fin: print(f"{indent}{f.split('/')[-1]}")

    # load all files inside dataset container
    da = xr.open_mfdataset(fin,
                           compat='override',
                           coords='minimal')[key]

    # change back variable name
    da = da.rename('PSL') if (key == 'msl') else da

    # rename coordinates
    if 'latitude' in [*da.coords.keys()]:

        da = da.rename({'valid_time': 'time',
                        'latitude'  : 'lat',
                        'longitude' : 'lon'})

    # check if lat coordinate should be sorted
    if (da['lat'][0] > da['lat'][-1]):

        da = da.sortby('lat')

    # check if lon coordinate should be sorted
    if (da['lon'].min() < 0):

        da['lon'] = da['lon'].where(da['lon'] >= 0, da['lon'] + 360)

        # sort longitude
        da = da.sortby('lon')

    # round spatial coordinates
    da['lat'] = da['lat'].astype(np.float32)
    da['lon'] = da['lon'].astype(np.float32)

    # reorder dimensions
    da = da.transpose(..., 'lon', 'lat')

    # make longitude cyclic
    if cyclic and not ():

        # clone first value to last position
        dalast        = da.sel({'lon': 0}).copy()
        dalast['lon'] = 360
        da            = xr.concat((da, dalast), 'lon')

    # output
    return da


def load_results(dirin, simid, dtype='cyclones', ctype = 'dict'):
    '''
    info:
    parameters:
    returns:
    '''

    # filenames for stored results
    fin  = (     f'{dtype}_{simid}_0001_0035.pkl' if ('ra' not in simid)
            else f'{dtype}_{simid}_1990_2001.pkl')

    # load cyclone tracks of simulation
    with open(f'{dirin}/{fin}', 'rb') as f:

        # read file
        data = pickle.load(f)

    # convert dictionary to list
    if ctype == 'list':

        # create new to arrangement for cyclone tracks information
        data2 = []

        # process each entry
        for cid in data.keys():

            # process each point
            for p in data[cid]:

                # only add new column if track data
                if dtype == 'tracks':

                    data2 += [[*p, cid]]

                else:

                    data2 += [p]

        # clean variable (to reduce memory usage)
        data = data2
        del data2

    # output
    return data


def convert_to_lambert(da, res=90, method='cubic'):
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
                           method=method)

        # output
        return da_k_xy

    # reorder dimensions
    da = da.copy().transpose(..., 'lon', 'lat')

    # create lat-lon grids (da.shape = [nt, nlat, nlon])
    LON, LAT = np.meshgrid(da['lon'].data, da['lat'].data, indexing='ij')

    # create transformer
    latlon_to_lambert = pyproj.Transformer.from_crs(latlon_crs,
                                                    laea_crs,
                                                    always_xy=True)

    lambert_to_latlon = pyproj.Transformer.from_crs(laea_crs,
                                                    latlon_crs,
                                                    always_xy=True)

    # transform curvilinear grid (lat-lon) to cartesian grid (x-y)
    X, Y = latlon_to_lambert.transform(LON, LAT)

    # define new regular x-y grid
    limx = np.floor(np.abs(X.max() / 100)) * 100
    limy = np.floor(np.abs(Y.max() / 100)) * 100

    # convert resolution to meters (from km)
    res *= 1000

    dx = (1) * res
    dy = (0) * res

    xn     = np.arange(-(limx+dx), (limx+dx)+1, res)
    yn     = np.arange(-(limy+dy), (limy+dy)+1, res)
    XN, YN = np.meshgrid(xn, yn, indexing='ij')

    # transform cartesian grid (x-y) to curvilinear grid (lat-lon)
    LON, LAT = lambert_to_latlon.transform(XN, YN)

    # check if time coordinate exists
    if len(da.shape) > 2:

        # list of dimensions
        dims = ['time', 'x', 'y']

        # define coords for xarray container
        coords = {'time': da['time'],
                  'x'   : xn,
                  'y'   : yn,
                  'lon' : (('x', 'y'), LON),
                  'lat' : (('x', 'y'), LAT)}

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

    else:

        # list of dimensions
        dims = ['x', 'y']

        # define coords for xarray container
        coords = {'x'  : xn,
                  'y'  : yn,
                  'lon': (('x', 'y'), LON),
                  'lat': (('x', 'y'), LAT)}

        # compute processing
        newarr = regrid_timestep(da)

    # make new container for regular x-y data
    nda = xr.DataArray(data=newarr,
                       dims=dims,
                       coords=coords,
                       attrs=da.attrs)

    # assign x-y grid coordinates to new data
    nda = nda.assign_coords({'X': (('x', 'y'), XN),
                             'Y': (('x', 'y'), YN)})

    # output
    return nda


def convert_to_lonlat(da, lon ,lat, method='cubic'):
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

        # retrieve points coordinates
        lonp = LON.flatten()
        latp = LAT.flatten()
        arrp = da_k.data.flatten()

        # create mask that points non-nans
        mask = ~(np.isnan(lonp) | np.isnan(latp) | np.isnan(arrp))

        # remove nans
        lonp = lonp[mask]
        latp = latp[mask]
        arrp = arrp[mask]

        # project data
        da_k_lonlat = griddata(points=(lonp, latp),
                               values=arrp,
                               xi=(LONR, LATR),
                               method=method)

        # output
        return da_k_lonlat

    # reorder dimensions
    da = da.copy().transpose(..., 'x', 'y')

    # create x-y grids (da.shape = [nt, nlat, nlon])
    X, Y = np.meshgrid(da['x'].data, da['y'].data, indexing='ij')

    # create transformer
    lambert_to_latlon = pyproj.Transformer.from_crs(laea_crs,
                                                    latlon_crs,
                                                    always_xy=True)

    # transform curvilinear grid (lat-lon) to cartesian grid (x-y)
    LON, LAT = lambert_to_latlon.transform(X, Y)

    # define new regular lon-lat grid
    LONR, LATR = np.meshgrid(lon, lat, indexing='ij')

    # change range of longitude
    LON = (LON + 360) % 360

    # check if time coordinate exists
    if len(da.shape) > 2:

        # list of dimensions
        dims = ['time', 'lon', 'lat']

        # define coords for xarray container
        coords = {'time': da['time'],
                  'lon' : lon,
                  'lat' : lat}

        # create container of projected data
        newarr = np.zeros((len(da['time']), len(lon), len(lat)))

        # create arguments iterator
        args = xarray_time_iterator(da)

        # create thread pool
        with mp.Pool(processes=25) as pool:

            # compute processing
            results = pool.imap(regrid_timestep, args)

            # process each timestep
            for k, da_k_lonlat in enumerate(results):

                # add results to container
                newarr[k, ::] = da_k_lonlat

    else:

        # list of dimensions
        dims = ['lon', 'lat']

        # define coords for xarray container
        coords = {'lon': lon,
                  'lat': lat}

        # compute processing
        newarr = regrid_timestep(da)

    # make new container for regular x-y data
    nda = xr.DataArray(data=newarr,
                       dims=dims,
                       coords=coords,
                       attrs=da.attrs)

    # assign x-y grid coordinates to new data
    nda = nda.assign_coords({'LON': (('lon', 'lat'), LONR),
                             'LAT': (('lon', 'lat'), LATR)})

    # output
    return nda


def distance_between_points(p1, p2):
    '''
    info:
    parameters:
    returns:
    '''

    # separate coordinates
    x1 = p1[0]
    x2 = p2[0]
    y1 = p1[1]
    y2 = p2[1]

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
    lambert_to_latlon = pyproj.Transformer.from_crs(laea_crs,
                                                    latlon_crs,
                                                    always_xy=True)

    # separate coordinates and value
    x = point[0]
    y = point[1]

    # project to lat-lon
    lon, lat = lambert_to_latlon.transform(x, y)

    # check if conversion is needed
    if type(lat) in (list, tuple):

        # convert longitude to array
        lon = np.array(lon)

    # transform lon range (from [-180, 180] to [0, 360])
    lon = ((lon + 360) % 360)

    # go back to list / tuple
    if type(lat) in (list, tuple):

        # convert longitude to list
        lon = [*lon]

    # output
    return [lon, lat]


def convert_point_to_xy(point):
    '''
    info:
    parameters:
    returns:
    '''


    # create transformer
    latlon_to_lambert = pyproj.Transformer.from_crs(latlon_crs,
                                                    laea_crs,
                                                    always_xy=True)

    # separate coordinates and value
    lon = point[0]
    lat = point[1]

    # transform point to cartesian grid (x-y)
    x, y = latlon_to_lambert.transform(lon, lat)

    # output
    return [x, y]


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


def add_trackpoints(xp, yp, npts=10):
    '''
    info:
    parameters:
    returns:
    '''

    # matrix with points
    points = np.array([xp, yp]).T

    # calculate distance from origin
    dist = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1)))
    dist = np.insert(dist, 0, 0)

    # new distances to get new points
    ndist = np.linspace(*dist[[0, -1]], npts)

    # method to use
    if len(dist) > 3:

        # interpolation method
        method = 'cubic'

    else:

        # interpolation method
        method = 'linear'


    if np.any(points[1:] == points[:-1]):

        return (xp, yp)

    # add new points to track
    interpolator = interp1d(dist, points, kind=method, axis=0)
    npoints      = interpolator(ndist)

    # separate new points
    nxp = npoints[:, 0]
    nyp = npoints[:, 1]

    # output
    return (nxp, nyp)


def add_trackpoints2(xp, yp, x, y):
    '''
    info:
    parameters:
    returns:
    '''

    # create containers for new points
    xnew = []
    ynew = []

    # interpolate per section
    for i in range(len(xp)-1):

        # length of section
        delta_xp = np.abs(xp[i+1] - xp[i])
        delta_yp = np.abs(yp[i+1] - yp[i])

        # check which coordinate to use as independent variable
        if (delta_xp) >= (delta_yp):

            # create mask with new interpolation points
            mask = (x >= np.min(xp[i:i+2])) & (x < np.max(xp[i:i+2]))

            # define new points
            xnewi = x[mask]

            # check order
            if (xp[i+1] < xp[i]):

                # reorder to decrease
                xnewi = xnewi[::-1]

            # create interpolation function
            f = interp1d(xp[i:i+2], yp[i:i+2])

            # interpolate
            ynewi = f(xnewi)

        else:

            # create mask with new interpolation points
            mask = (y >= np.min(yp[i:i+2])) & (y < np.max(yp[i:i+2]))

            # define new points
            ynewi = y[mask]

            # check order
            if (yp[i+1] < yp[i]):

                # reorder to decrease
                ynewi = ynewi[::-1]

            # create interpolation function
            f = interp1d(yp[i:i+2], xp[i:i+2])

            # interpolate
            xnewi = f(ynewi)

        # add new points to containers
        xnew += [*xnewi]
        ynew += [*ynewi]

    # convert containers to arrays
    xnew = np.array(xnew)
    ynew = np.array(ynew)

    # output
    return (xnew, ynew)










