#!/home/cr2/agomez/.conda/envs/conda/bin/python
# -*- coding: utf-8 -*-
#
# ~/projects/piss/src/piss_map_01_slp_minimums.py
#
# info  : plot maps with slp minimums locations
# author: @alvaroggc


# standard libraries
import copy
import csv
import datetime as dt
import gc
import locale
import os
import pickle
import re
import sys
# import warnings
from glob import glob

# 3rd party packages
import cf
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import multiprocess as mp
import matplotlib.path as mpath
import matplotlib.pyplot as plt
from pyproj import Geod
from shapely.geometry import Point, LineString
from shapely.geometry.polygon import Polygon

# local source
from _piss_lib import *

# figures configuration
plt.rcParams['figure.dpi'     ] = 200
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8

# general configuration
# warnings.filterwarnings('ignore')               # supress deprecation warnings
locale.setlocale(locale.LC_ALL, 'es_CL.UTF-8')  # apply spanish locale settings


###########################
##### LOCAL FUNCTIONS #####
###########################


def get_variables(args: list[str]):
    '''
    info: retrieve variables from input arguments.
    parameters:
        args : list[str] -> list of arguments to parse.
    returns:
        simid : str  -> list of simulation(s) name(s) (default is 'lgm_100').
    '''

    # get copy of arguments
    args = args[1:].copy()

    # formats
    simid_fmt = re.compile(r'^lgm_\d{3}?')      # simulation id
    ra_fmt    = re.compile(r'^ra_(6|24)h$')

    # retrieve variables
    simids = [arg for arg in args if simid_fmt.match(str(arg))]   # sim. ID
    simids = [arg for arg in args if ra_fmt.match(str(arg))]

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
dirsim  = f'/mnt/cirrus/results/friquelme'      # cesm simulations
dirdata = f'{homedir}/projects/piss/data'       # data output
dirimg  = f'{homedir}/projects/piss/img/slpmins'       # output for figures

# output file with cyclones information
fin = f'cyclones_{simid}_v3_0001_0035.pkl'

# indexer to choose southern hemisphere (from 30°S to the south)
shidx = {'lat': slice(-90, -30)}

# figure parameters
size = (7.5, 6)


###################
##### RUNNING #####
###################


# read file
cyclones = load_results(dirdata, simid, dtype='cyclones', ctype='dict')

# temporal range
date_ini = [*cyclones.keys()][ 0]
date_end = [*cyclones.keys()][-1]

# load simulation datasets
slp = load_simulation_variable(dirsim, simid, 'PSL', cyclic=True)

# extract temporal range (only for slp)
slp = slp.sel({'time': slice(date_ini, date_end)})

# interpolate data to CESM resolution (only for reanalysis)
if (simid == 'ra'):

    topo = load_simulation_variable(dirsim, simid, 'PHIS')

    slp = slp.interp({'lat': topo['lat'],
                      'lon': topo['lon']}, method='linear')

    del topo

# resample data (only for ra dataset)
if (simid == 'ra_24h'):

    # from 6h resolution to 24h res.
    slp = slp.isel({'time': slice(None, None, 4)})

# adjust slp units
slp = slp.where(False, slp / 100)
slp.attrs['units'] = 'hPa'

# output filename template
output_template = f'piss_map_slpmins_{simid.lower()}_*.png'

# remove previous images
for f in glob(f'{dirimg}/{output_template}'): os.remove(f)

# levels of slp (for maps)
levels = np.arange(950, 1050+1, 5)

# axis parameters to make it circular
theta  = np.linspace(0, 2*np.pi, 100)
center = [0.5, 0.5]
radius = 0.5
verts  = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)

# logging message
indent = log_message('making slpmin maps')

# plot tracks over map
for t in slp['time'][:25]:

    # date identifiers
    datestr = t.dt.strftime('%Y-%m-%d %H:%M').item()
    dateid  = t.dt.strftime( '%Y%m%d%H%M' ).item()

    # output filename image
    output = f"{output_template.replace('*', dateid)}"

    # logging message
    print(f"{indent}{output.split('/')[-1]}")

    # extract coordinates of track points and values
    xp   = np.array([p[ 0] for p in cyclones[datestr]])
    yp   = np.array([p[ 1] for p in cyclones[datestr]])
    cidp = np.array([p[-1] for p in cyclones[datestr]])

    # convert units (from km to meters)
    xp *= 1000
    yp *= 1000

    # time indexer
    tidx = {'time': t}

    # create figure
    fig, ax = plt.subplots(1, 1, figsize=size,
                           subplot_kw={'projection': proj})

    # plot field
    slp.sel(tidx).plot.contourf(ax=ax,
                                transform=trans,
                                levels=levels,
                                extent='both',
                                cmap='jet',
                                cbar_kwargs={'location'   : 'right',
                                                'orientation': 'vertical',
                                                'drawedges'  : False,
                                                'fraction'   : 0.1,
                                                'shrink'     : 1,
                                                'aspect'     : 30,
                                                'pad'        : 0.00,
                                                'anchor'     : (0.5, 1),
                                                'panchor'    : (0.5, 0)})

    # draw points in map
    ax.plot(xp, yp, transform=transxy,
            lw=1, ls='', marker='o', ms=11, mfc='yellow', mec='black')

    # plot contours and codes of points
    for ip, p in enumerate(cyclones[datestr]):

        # separate contour coordinates
        xc, yc = p[4][1:]

        # convert units (from km to meters)
        xc *= 1000
        yc *= 1000

        # write code of point
        ax.text(xp[ip], yp[ip], cidp[ip], transform=transxy,
                size=5, color='black', alpha=0.8, ha='center', va='center')

        # draw contour
        ax.plot(xc, yc, lw=1, color='black', transform=transxy)

    # add coastlines
    ax.coastlines(color='black', linewidth=0.5)

    # add and adjust gridlines
    gridlines = ax.gridlines(linewidth=1,
                             color='grey',
                             alpha=0.25,
                             ls='--',
                             draw_labels=True,
                             x_inline=False,
                             y_inline=False)

    gridlines.top_labels    = False
    gridlines.bottom_labels = False
    gridlines.left_labels   = True
    gridlines.right_labels  = True

    # set labels
    ax.set_title(f'Southern Hemishphere, {datestr}')
    ax.set_xlabel('')
    ax.set_ylabel('')

    # remove box
    # ax.axis('off')

    # make axis circular
    ax.set_boundary(circle, transform=ax.transAxes)

    # set coordinate limits
    ax.set_extent([0, 360, -90, -20], crs=trans)

    # save / show plot
    fig.savefig(f"{dirimg}/{output}", bbox_inches='tight')

    # close figure
    plt.close()

