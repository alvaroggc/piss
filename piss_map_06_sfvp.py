#!/home/cr2/agomez/.conda/envs/conda/bin/python
# -*- coding: utf-8 -*-
#
# ~/projects/piss/src/piss_map_06_sfvp.py
#
# info  : show stream function (sf) or potential velocity (vp) climatology
#         over the southern hemisphere
# usage : ./piss_map_06_sfvp.py <simid>
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
from functools import partial
from glob import glob

# 3rd party packages
import cartopy.crs as ccrs
import cf
import colormaps as cmaps
import dask
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




def smooth_array(da, r):
    '''
    info:
    parameters: da -> xr.DataArray : (lat|lon x lon|lat) array container
    returns:
    '''

    # create filter
    y, x = np.ogrid[-r:r+1, -r:r+1]
    disk = x**2 + y**2 <= r**2
    disk = disk.astype(float)
    disk = disk / disk.sum()

    # smooth 2d-array
    arr = signal.convolve2d(da.data, disk, mode='same', boundary='wrap')

    # replace values
    da = da.where(False, arr)

    # output
    return da


def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    author(ess): @paul-h
    url: https://stackoverflow.com/questions/7404116/defining-the-midpoint-of-a-colormap-in-matplotlib

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False),
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap


def get_variables(args: list[str]):
    '''
    info: retrieve variables from input arguments.
    parameters:
        args : list[str] -> list of arguments to parse.
    returns:
        simid : str  -> list of simulation(s) name(s) (default is 'lgm_100').
        plot  : bool -> plot maps with cyclones
        calc  : bool -> search for cyclones
    '''

    # get copy of arguments
    args = args[1:].copy()

    # formats
    simid_fmt = re.compile(r'^lgm_\d{3}?')      # simulation id
    # ra_fmt    = re.compile(r'^ra_(6|24)h$')

    # retrieve variables
    simid = [arg for arg in args if simid_fmt.match(str(arg))]   # sim. ID
    # simid = [arg for arg in args if ra_fmt.match(str(arg))]
    anual = True if 'anual' in args else False
    anom  = 'a'  if 'anom'  in args else ''

    # check arguments
    simid = 'lgm_100' if not simid else simid[0]

    # output
    return (simid, anual, anom)


############################
##### LOCAL PARAMETERS #####
############################


# get variables from input arguments
simid, anual, anom = get_variables(sys.argv)

# directories
homedir = os.path.expanduser('~')               # home directory
projdir = f'{homedir}/projects/piss'
svrdir  = '/mnt/arcus/results/agomez/projects/piss'

dirsim  = f'/mnt/cirrus/results/friquelme'      # cesm simulations
dirdata = f'{svrdir}/data/sfvp'                # data output
dirimg  = f'{projdir}/img/sfvp'    # output for figures

# variable to load
key = 'SF850'

# start date (first 5 years are transitional)
date_start = cf.dt('0005-01-01 00:00:00', calendar='noleap')

# indexer to choose southern hemisphere (from -30° to the south)
shidx = {'lat': slice(-90, -30)}

# smooth radius
r = 2

# figure parameters
size = (15, 12)

# extra parameters
tab = (" " * 4)


###################
##### RUNNING #####
###################


# logging message
print(f'\n:: [{simid}] creating stream function maps')

# update output directory and create if it doesn't exists
os.makedirs(f'{dirimg}') if not os.path.isdir(f'{dirimg}') else None

# climatology filename
if anual:

    fin  = f'{simid}.timmean.sfvp.nc'
    finb = f'lgm_100.timmean.sfvp.nc'

else:

    fin  = f'{simid}.yseasmean.sfvp.nc'
    finb = f'lgm_100.yseasmean.sfvp.nc'

# load base simulation
da   = xr.open_dataset(f'{dirdata}/{fin}',  use_cftime=True)[key]
base = xr.open_dataset(f'{dirdata}/{finb}', use_cftime=True)[key]

# select region
da   =   da.sel(shidx)
base = base.sel(shidx)

# calculate anomaly
if anom:

    da = da.where(False, (da.data - base.data))

    # update attributes
    da.attrs['long_name'] += ' (anomaly)'

    # set limits
    vmin = -4e+6
    vmax =  4e+6

else:

    # set limits
    vmin = 0
    vmax = 1.5e+8

# # smooth fields
# da = smooth_array(da, r)

# colorbar arguments
colorbar_kwargs = {
                   # 'ticks' : levels[::3],
                   # 'shrink': 0.75,
                   'orientation': 'vertical',
                   # 'pad'        : -1,
                   # 'aspect'     : 30,
                   # 'shrink'     : 0.3,
                   }

# parameters to shape axis as circle
theta  = np.linspace(0, 2*np.pi, 100)
center = [0.5, 0.5]
radius = 0.5
verts  = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)

# parameters for labels
# period = ['summer', 'autumn', 'winter', 'spring']
period = ['DEF', 'MAM', 'JJA', 'SON']

# define crop command
crop_cmd = '/usr/bin/convert -crop 1250x900+120+0'

# process each timestep
for i, t in enumerate(da['time']):

    # create figure
    fig, ax = plt.subplots(1, 1, figsize=size, subplot_kw={'projection': proj})

    # time indexer
    idx = {'time': t}

    # plot field
    da.sel(idx).plot.pcolormesh(ax=ax,
                                transform=trans,
                                x='lon',
                                vmin=vmin,
                                vmax=vmax,
                                # levels=levels,
                                extend='both',
                                add_colorbar=True,
                                cmap='RdBu_r',
                                cbar_kwargs=colorbar_kwargs)

    # add coastlines
    ax.coastlines(color='black', lw=0.5, ls='-')

    # define labels
    if anual:

        # title
        title = f'{simid.upper()}, clim. annual (0005 - 0035)'

        # output image filename
        fout = f'piss_map_{key.lower()}{anom}_annual_{simid}.png'

    else:

        # title
        title = f'{simid.upper()}, clim. {period[i]} (0005 - 0035)'

        # output image filename
        fout = f'piss_map_{key.lower()}{anom}_{period[i].lower()}_{simid}.png'

    # add final detail to title
    title += f' [{anom}]' if anom else ''

    # add and adjust gridlines
    gridlines = ax.gridlines(linewidth=1,
                             color='grey',
                             alpha=0.1,
                             ls='--',
                             draw_labels=False,
                             x_inline =False,
                             y_inline =False)

    # gridlines.xlocator = mticker.MultipleLocator(30)

    gridlines.top_labels    = False
    gridlines.bottom_labels = False
    gridlines.left_labels   = True
    gridlines.right_labels  = False

    # set labels
    ax.set_title(title)
    ax.set_xlabel('')
    ax.set_ylabel('')

    # make axis circular
    ax.set_boundary(circle, transform=ax.transAxes)

    # set coordinate limits
    ax.set_extent([0, 360, -90, -30], crs=trans)

    # logging message
    print(f'{tab}{fout}')

    # save / show plot
    # fig.tight_layout()
    fig.savefig(f"{dirimg}/{fout}", bbox_inches='tight')
    plt.close()

    # # crop temporal pdf to remove unnecessary headers
    # os.system(f'{crop_cmd} {dirimg}/{fout} {dirimg}/{fout} > /dev/null')


# final logging message
print('\n:: uwu\n\n')


























