#!/home/alvaro/.conda/envs/conda/bin/python
# -*- coding: utf-8 -*-
#
# ~/projects/piss/src/piss_pproc_loncross.py
#
# info  : <something>
# author: @alvaroggc


# standard libraries
import copy
import csv
import datetime as dt
import locale
import os
import pickle
import re
import sys
import itertools

from glob import glob
from functools import partial

# 3rd party packages
import cf
import colormaps as cmaps
import pyproj
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import multiprocess as mp
import numpy as np
import xarray as xr

from scipy import signal
from scipy.interpolate import (griddata, interp1d)

# local source
from _piss_lib import *

# figures configuration
plt.rcParams['figure.dpi'     ] = 200
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8

# general configuration
# warnings.filterwarnings('ignore')             # supress deprecation warnings
locale.setlocale(locale.LC_ALL, 'es_CL.UTF-8')  # apply spanish locale settings


###########################
##### LOCAL FUNCTIONS #####
###########################


def expand_pismask(pismask, D=840000):
    '''
    info:
    parameters:
    returns:
    '''

    # create copy of pismask with only zeros
    newpismask = pismask.where(False, 0)

    # process each grid point
    for i in range(len(pismask['x'])):

        # extract y-coordinate of gridpoint
        xi = pismask['x'][i].item()

        for j in range(len(pismask['y'])):

            # extract y-coordinate of gridpoint
            yj = pismask['y'][j].item()

            # skip if gridpoint not in pis
            if (pismask.sel({'x': xi, 'y': yj}).item() == 0):

                # continue to next point
                continue

            # create mask with distances from gridpoint
            dist = distance_between_points((xi, yj), (pismask['X'].data,
                                                      pismask['Y'].data))

            # mask with points outside search radius
            mask = (dist > D)

            # expand pismask (fill with ones inside radius)
            newpismask = newpismask.where(mask, 1)

    # output
    return newpismask


############################
##### LOCAL PARAMETERS #####
############################


# directories
homedir = os.path.expanduser('~')               # home directory
projdir = f'{homedir}/projects/piss'
svrdir  = f'/mnt/arcus/results/agomez/projects/piss'

dirsim  = f'/mnt/cirrus/results/friquelme'      # cesm simulations
dirout  = f'{projdir}/data/cross'      # cesm simulations

dirdata     = f'{projdir}/data'      # cesm simulations
dirdata_svr = f'{svrdir}/data'      # cesm simulations

# list of all simulation names
simids = ['lgm_050', 'lgm_100', 'lgm_150', 'lgm_200', 'lgm_300']

# seasons
seasons_list = ['DEF', 'MAM', 'JJA', 'SON']

# range of latitudes
lat_bands = {
             'N' : (-45, -38),
             'C' : (-50, -45),
             'S' : (-56, -50),
             'D1': (-62, -56),
             'D2': (-70, -62)
             }

lat_bands_names = {'N' : 'North PIS',
                   'C' : 'Central PIS',
                   'S' : 'Southern PIS',
                   'D1': 'Drake Passage (north)',
                   'D2': 'Drake Passage (south)'}

# longitude range
lon_min = -90
lon_max = -30

tab = (" " * 4)


###################
##### RUNNING #####
###################


# create transformer
latlon_to_lambert = pyproj.Transformer.from_crs(latlon_crs,
                                                laea_crs,
                                                always_xy=True)

lambert_to_latlon = pyproj.Transformer.from_crs(laea_crs,
                                                latlon_crs,
                                                always_xy=True)

# create regular lat-lon coordinates
res = 0.85

lonr = np.arange(-180, 180, res)
latr = np.arange( -90,   0, res)

LONR, LATR = np.meshgrid(lonr, latr, indexing='ij')

# logging message
print('extracting cross sections')

# contourf levels to color
levels = np.arange(0, 20+1, 2)

# colormap
cmap = cmaps.WhiteBlueGreenYellowRed

for i, season in enumerate(seasons_list):

    if season not in ['DEF', 'JJA']:
    # if season not in ['DEF']:

        continue

    # process each simulation
    for simid in simids:

        # filename
        fin_count = f'cyclonic_ocurrences_{simid}_{season.lower()}.nc'
        fin_sf    = f'{simid}.yseasmean.sfvp.nc'

        # load base simulation
        count = xr.open_dataset(f'{dirdata}/{fin_count}')['count']
        sf200 = xr.open_dataset(f'{dirdata_svr}/sfvp/{fin_sf}')['SF200']
        sf850 = xr.open_dataset(f'{dirdata_svr}/sfvp/{fin_sf}')['SF850']

        sf200['lon'] = sf200['lon'].where((sf200['lon'].data < 180), (sf200['lon'].data - 360))
        sf850['lon'] = sf850['lon'].where((sf850['lon'].data < 180), (sf850['lon'].data - 360))

        sf200 = sf200.isel({'time': i})
        sf850 = sf850.isel({'time': i})

        sf200 = sf200.sortby('lon')
        sf850 = sf850.sortby('lon')

        # extract coordinates
        x = count['x'].data
        y = count['y'].data

        lon = count['lon'].data
        lat = count['lat'].data

        # mask without nans
        mask = (  ( ~np.isnan(count.data.flatten()) )
                & ( ~np.isnan(       lon.flatten()) )
                & ( ~np.isnan(       lat.flatten()) )  )

        # alias to project data
        project = partial(griddata,
                          points=(lon.flatten()[mask], lat.flatten()[mask]),
                          xi=(LONR, LATR),
                          method='cubic')

        # project data
        count_latlon_arr = project(values=count.data.flatten()[mask])

        # create xarray containers
        count = xr.DataArray(data=count_latlon_arr,
                             name='count',
                             coords={'lon': lonr,
                                     'lat': latr},
                             dims=['lon', 'lat'],
                             attrs=count.attrs)

        # # create figure
        # fig, ax = plt.subplots(1, 1, figsize=(10, 10), subplot_kw={'projection': proj})
        #
        # ax.contourf(LONR, LATR, count_latlon_arr,
        #             transform=trans,
        #             levels=levels,
        #             cmap=cmap)
        #
        # # add coastlines
        # ax.coastlines(color='black', lw=0.5)
        #
        # # add and adjust gridlines
        # gridlines = ax.gridlines(linewidth=1,
        #                          color='grey',
        #                          alpha=0.1,
        #                          ls='--',
        #                          draw_labels=False,
        #                          x_inline =False,
        #                          y_inline =False)
        #
        # # gridlines.xlocator = mticker.MultipleLocator(30)
        # gridlines.top_labels    = False
        # gridlines.bottom_labels = False
        # gridlines.left_labels   = True
        # gridlines.right_labels  = False
        #
        # # set labels
        # title = f'{simid.upper()}'
        #
        # ax.set_title(title)
        # ax.set_xlabel('')
        # ax.set_ylabel('')
        #
        # fig.savefig(f'{homedir}/test.png')
        # plt.close()

        # logging message
        print('')

        # process each latitude band
        for band in lat_bands:

            # extract latitude limits
            lat_min = lat_bands[band][0]
            lat_max = lat_bands[band][1]

            # spatial indexer
            idx = {'lon': slice(lon_min, lon_max),
                   'lat': slice(lat_min, lat_max)}

            # get cross section of band
            count_cross_avg = count.sel(idx).mean(dim='lat', skipna=True, keep_attrs=True)
            count_cross_max = count.sel(idx).max( dim='lat', skipna=True, keep_attrs=True)

            sf200_cross = sf200.sel(idx).mean(dim='lat', skipna=True, keep_attrs=True)
            sf850_cross = sf850.sel(idx).mean(dim='lat', skipna=True, keep_attrs=True)

            # interpolate stream function variables
            sf200_cross = sf200_cross.interp({'lon': count_cross_avg['lon']}, method='cubic')
            sf850_cross = sf850_cross.interp({'lon': count_cross_avg['lon']}, method='cubic')

            # filename
            fout = f'{simid.lower()}.piss_{band.lower()}.{season.lower()}.cross.nc'
            fout = f'{dirout}/{fout}'

            # logging message
            print(f"({season}, {band}) {fout.split('/')[-1]}")

            # encoding options
            encoding = {'count_avg': {'zlib': True, 'complevel': 5},
                        'count_max': {'zlib': True, 'complevel': 5},
                        'SF200'    : {'zlib': True, 'complevel': 5},
                        'SF850'    : {'zlib': True, 'complevel': 5},}

            # save data
            xr.Dataset({'count_avg': count_cross_avg,
                        'count_max': count_cross_max,
                        'SF200': sf200_cross,
                        'SF850': sf850_cross}).to_netcdf(fout,
                                                         encoding=encoding)


















