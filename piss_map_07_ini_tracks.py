#!/home/cr2/agomez/.conda/envs/conda/bin/python
# -*- coding: utf-8 -*-
#
# ~/projects/piss/src/piss_map_07_ini_tracks.py
#
# info  : show tracks beggining of cyclonic activity over southern hemisphere
#         in piss simulations (cesm2), based on Hanley & Caballero (2012).
# usage : ./piss_map_07_ini_tracks.py [simid]
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
import dask
import pyproj
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import xarray as xr
from matplotlib.patches import Arc

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
    for i in range(len(pismask['y'])):

        # extract y-coordinate of gridpoint
        yi = pismask['y'][i].item()

        for j in range(len(pismask['x'])):

            # extract y-coordinate of gridpoint
            xj = pismask['x'][j].item()

            # skip if gridpoint not in pis
            if (pismask.sel({'y': yi, 'x': xj}).item() == 0):

                # continue to next point
                continue

            # create mask with distances from gridpoint
            dist = distance_between_points((xj, yi), (pismask['X'].data,
                                                      pismask['Y'].data))

            # mask with points outside search radius
            mask = (dist > D)

            # expand pismask (fill with ones inside radius)
            newpismask = newpismask.where(mask, 1)

    # output
    return newpismask


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
    simid_fmt  = re.compile(r'^lgm_\d{3}?')      # simulation id
    raid_fmt   = re.compile(r'^ra_(6|24)h$')
    pis_sa_fmt = re.compile(r'^(pis|sa)$')
    D_fmt      = re.compile(r'^D\d')             # search distance

    # retrieve variables
    simid = [arg for arg in args if simid_fmt.match(str(arg))]   # sim. ID
    raid  = [arg for arg in args if raid_fmt.match(str(arg))]

    simid = raid if raid else simid

    pis_sa = [f'_{arg}' for arg in args if pis_sa_fmt.match(str(arg))]

    nocoast = '_nocoast' if 'nocoast' in args else ''

    # check arguments
    simid = 'lgm_100' if not simid  else simid[0]
    pis_sa = ''       if not pis_sa else pis_sa[0]

    # output
    return (simid, pis_sa, nocoast)


############################
##### LOCAL PARAMETERS #####
############################


# get variables from input arguments
simid, pis_sa, nocoast = get_variables(sys.argv)

# directories
homedir = os.path.expanduser('~')                       # home directory
dirsim  = f'/mnt/cirrus/results/friquelme'              # cesm simulations
dirref  = f'{homedir}/projects/thesis/postprocessing/extra'  # cesm simulations
dirdata = f'{homedir}/projects/piss/data'               # data output
dirimg  = f'{homedir}/projects/piss/img/tracks_ini/{nocoast[1:]}'

# start date (first 5 years are transitional)
date_start = (     cf.dt('1990-01-01 00:00:00', calendar='gregorian') if ('ra' in simid)
              else cf.dt('0005-01-01 00:00:00', calendar='noleap'))

# indexer to choose southern hemisphere (from 30°S to the south)
shidx = {'lat': slice(-90, -30)}

# figure parameters
size = (7.5, 6) if (not 'pis' in pis_sa) else (15, 12)


###################
##### RUNNING #####
###################


# open patagonian ice sheet spatial mask
pismask = xr.open_dataset(f'{dirdata}/pis_mask.nc')['ice']

# fix longitude to be cyclic
lon = pismask['lon'].data
lon[-1] = 360
pismask['lon'] = pismask['lon'].where(False, lon)

# extract southern hemisphere
pismask = pismask.sel(shidx)

# convert coordinate system (from lat-lon to x-y)
pismask = convert_to_lambert(pismask, method='nearest')

# convert coordinates units (from km to m) (only 1D)
pismask['x'] = pismask['x'].where(False, pismask['x'] * 1000)
pismask['y'] = pismask['y'].where(False, pismask['y'] * 1000)

pismask['X'] = pismask['X'].where(False, pismask['X'] * 1000)
pismask['Y'] = pismask['Y'].where(False, pismask['Y'] * 1000)

# extract x-y coordinates (1D)
x = pismask['x'].data
y = pismask['y'].data

# expand pismask
pismask_expanded = expand_pismask(pismask, D=840000)

# text box parameters
bbox = {'boxstyle' : 'round',
        'facecolor': 'wheat',
        'edgecolor': 'black',
        'alpha'    : 0.5}

# axis parameters to make it circular
if ('pis' in pis_sa):

    theta_min = -np.pi / 4
    theta_max =  np.pi / 4

    theta  = np.linspace(theta_min, theta_max, 100)
    center = [0.5, 0.5]

    radius1 = 0.5
    radius2 = 0.17

    verts1  = np.vstack([np.sin(theta), np.cos(theta)]).T
    verts2  = np.vstack([np.sin(theta[::-1]), np.cos(theta[::-1])]).T

    # Añadir vértice del centro para cerrar el "pedazo de pizza"
    verts = np.vstack([verts1 * radius1 + center, verts2 * radius2 + center])

    circle = mpath.Path(verts)

    datestr_pos = [0.78, 0.62]


else:

    theta  = np.linspace(0, 2*np.pi, 100)
    center = [0.5, 0.5]
    radius = 0.5
    verts  = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)

    datestr_pos = [0.78, 0.0]

if ('sa' in pis_sa):

    datestr_pos = [0.78, -0.1]

# scale factor for pis grid markers
pis_sa_ms  = 1     if pis_sa else 0.5
pis_sa_res = '50m' if pis_sa else '110m'

# define crop command
crop_cmd = f'{homedir}/.conda/envs/conda/bin/convert -crop 1850x1000+240+0'

# update output directory and create if it doesn't exists
os.makedirs(f'{dirimg}') if not os.path.isdir(f'{dirimg}') else None

# load track results
tracks = load_results(dirdata, simid, dtype='tracks', ctype='dict')

# logging message
indent = log_message(f'[{simid}] creating storm track tracks map')

# output filename template
output = f'piss_map{pis_sa}_ini_tracks_{simid.lower()}{nocoast}.png'

# create figure
fig, ax = plt.subplots(1, 1, figsize=size, subplot_kw={'projection': proj})

# add coastlines
ax.coastlines(color='black',
              linewidth=0.5,
              resolution=pis_sa_res) if (not nocoast) else None

# create object for initial temporal range
date_ini = ''

# track counter
nt = 0

# type of calendar ot time coordinate
calendar = 'gregorian' if ('ra' in simid) else 'noleap'

# get tracks keys list
tids = [*tracks.keys()]

# extract points of first step of each track
xp = [tracks[tid][0][0] for tid in tids]
yp = [tracks[tid][0][1] for tid in tids]
tp = [cf.dt(tracks[tid][0][5], calendar=calendar) for tid in tids]

# convert to arrays
xp = np.array(xp)
yp = np.array(yp)
tp = np.array(tp)

# convert units (from km to meters)
xp *= 1000
yp *= 1000

# remove first 5 transitional years
mask5years = (tp < date_start)

xp = xp[~mask5years]
yp = yp[~mask5years]
tp = tp[~mask5years]

# get temporal range
date_ini = tp[ 0]
date_end = tp[-1]

# logging message
print(f'{indent}drawing first point of each track')

# plot first point of each track
ax.scatter(xp, yp, 20,
           transform=transxy,
           ls='-',
           alpha=0.05,
           fc='darkblue',
           ec='none')

# number of tracks considered
nt = len(tp)

# create temporal information box
datestr  = f"Years : {date_ini.strftime('%Y')} - {date_end.strftime('%Y')}"
datestr += f"\nSeason: year-round"

# # add temporal info. box
ax.text(*datestr_pos, datestr, bbox=bbox, size=7, family='monospace',
        transform=ax.transAxes)

# # add pis gridpoints
# ax.scatter(pismask_expanded['X'].data,
#            pismask_expanded['Y'].data,
#            pismask_expanded.data*0.5,
#            transform=transxy, c='black', alpha=0.05)
#
# # add pis gridpoints
# ax.scatter(pismask['X'].data,
#            pismask['Y'].data,
#            pismask.data*pis_sa_ms,
#            zorder=3,
#            transform=transxy, c='black', alpha=0.45) if (not nocoast) else None

# add and adjust gridlines
gridlines = ax.gridlines(linewidth=1,
                         color='grey',
                         # alpha=0.00,
                         alpha=0.1,
                         ls='--',
                         draw_labels=False,
                         x_inline=False,
                         y_inline=False)

gridlines.top_labels    = False
gridlines.bottom_labels = False
gridlines.left_labels   = True
gridlines.right_labels  = False

# gridlines.xlocator = mticker.FixedLocator([])
# gridlines.ylocator = mticker.FixedLocator([])
# gridlines.ylocator = mticker.FixedLocator([-56, -38])

# set labels
ax.set_title(f'{simid.upper()} (total = {nt:d})')
ax.set_xlabel('')
ax.set_ylabel('')

# logging message
print(f'{indent}{output}')

if (not pis_sa) or ('pis' in pis_sa):

    # make axis circular
    ax.set_boundary(circle, transform=ax.transAxes)

    # set coordinate limits
    ax.set_extent([0, 360, -90, -30], crs=trans)

else:

    # set coordinate limits [W, E, S, N]
    ax.set_extent([275, 310, -60, -30], crs=trans)

# save / show plot
fig.tight_layout() if (not 'sa' in pis_sa) else None
fig.savefig(f'{dirimg}/{output}', bbox_inches='tight')
plt.close()

# crop temporal pdf to remove unnecessary headers
if ('pis' in pis_sa):

    os.system(f'{crop_cmd} {dirimg}/{output} {dirimg}/{output} > /dev/null')

# final logging message
print('\n:: uwu\n\n')


























