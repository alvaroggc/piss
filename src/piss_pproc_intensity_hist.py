#!/home/cr2/agomez/.conda/envs/conda/bin/python
# -*- coding: utf-8 -*-
#
# ~/projects/piss/src/piss_pproc_intensity_hist.py.py
#
# info  : create intensity histogram of cyclone tracks
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

# 3rd party packages
import cf
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
from piss_lib import *

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


def grid_index_iterator(x, y):
    '''
    info:
    parameters:
    returns:
    '''

    # create grids from coordinates
    iX, iY = np.mgrid[:len(x), :len(y)]

    # flatten grids
    iX = iX.flatten()
    iY = iY.flatten()

    # process each gridpoint
    for ij in range(len(iX)):

        # generate iterator
        yield (iX[ij], iY[ij])


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


def get_pis_tracks(ij):
    '''
    info:
    parameters:
    returns:
    '''

    # extract coordinates of gridpoint
    latref = lat[ij[0]]
    lonref = lon[ij[1]]
    iceref = ice[ij[0], ij[1]].item()

    # if point not over pis, return empty list
    if iceref == 0:

        # output
        return []

    # calculate distance
    dist = distances_from_point(latref, lonref, latp, lonp)

    # mask with points inside search radius
    mask = (dist < D)

    # list with unique cyclone tracks ids
    valid_tracks = [*np.unique(cidp[mask])]

    # logging message
    print(f'{indent}({latref:5.1f}, {lonref:5.1f}) {len(valid_tracks):.0f}')

    # output
    return valid_tracks


############################
##### LOCAL PARAMETERS #####
############################


# directories
homedir = os.path.expanduser('~')               # home directory
projdir = f'{homedir}/projects/piss'

dirdata = f'{projdir}/data'      # cesm simulations
dirimg  = f'{projdir}/img/extra'        # output for figures

# path to reference grid file
refpath  = f'{homedir}/projects/thesis/postprocessing/extra/'
refpath += f'ra.era5.cam.sst.nc'

# output image filename
output = 'piss_hist_pis.png'

# list of all simulation names
simids = ['lgm_050', 'lgm_100', 'lgm_150', 'lgm_200', 'lgm_300']

# parameters for method
D = 840     # km

# start date (first 5 years are transitional)
date_start = cf.dt('0005-01-01 00:00:00', calendar='noleap')

# indexer to choose southern hemisphere (from 20°S to the south)
shidx = {'lat': slice(-90, -20)}

# figure parameters
size = (7.5, 6)

# text box parameters
bbox = {'boxstyle' : 'round',
        'facecolor': 'white',
        'edgecolor': 'black',
        'alpha'    : 1.0}


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

# create containers for results
msl_pis_min_lgm   = []
delta_pis_max_lgm = []

bins_msl   = np.arange(935, 1036, 10)
bins_delta = np.arange(7.5, 72.5,  5)

# logging message
indent = log_message('searching for intensity values')

# update output directory and create if it doesn't exists
os.makedirs(f'{dirimg}') if not os.path.isdir(f'{dirimg}') else None

# process each simulation
for simid in simids:

    # logging message
    print(f'{indent}{simid}')

    # load cyclone tracks of simulation
    tracks = load_results(dirdata, simid, dtype='tracks', ctype='dict')

    # create container for initial temporal range
    date_ini = []

    # logging message
    indent = log_message(f'[{simid}] checking pis cyclones')

    # create container for this simulation
    msl_pis_min   = []
    delta_pis_max = []

    # process each tracks
    for tid in tracks.keys():

        # separate tracks components
        yp     = np.array([p[ 0] for p in tracks[tid]])       # y-coordinate
        xp     = np.array([p[ 1] for p in tracks[tid]])       # x-coordinate
        tp     = cf.dt_vector([p[5] for p in tracks[tid]], calendar='noleap')  # t-coordinate
        mslp   = np.array([p[ 2] for p in tracks[tid]])       # sea level pressure
        deltap = np.array([p[ 3] for p in tracks[tid]])       # sea level pressure drop

        # convert units (from km to meters)
        xp *= 1000
        yp *= 1000

        # remove first 5 transitional years
        mask5years = (tp < date_start)

        # skip if data contains anything from first 5 years
        if ((~mask5years).sum() < len(mask5years)):

            # continue to next track
            continue

        # get temporal range
        date_ini = tp[ 0] if not date_ini else date_ini
        date_end = tp[-1]

        # add more points to track
        xp_interp, yp_interp = add_trackpoints2(xp, yp, x, y)

        # mask to check if track goes over pis
        inpis = np.array([pismask_expanded.sel(x=xp_interp[i],
                                               y=yp_interp[i],
                                               method='nearest').item()
                          for i in range(len(xp_interp))])

        # if no point crosses pis, skip
        if np.sum(inpis[~np.isnan(inpis)]) == 0:

            # go to next track
            continue

        # add data to containers
        msl_pis_min   += [mslp.min()]
        delta_pis_max += [deltap.max()]

    # convert results to arrays
    msl_pis_min   = np.array(msl_pis_min  )
    delta_pis_max = np.array(delta_pis_max)

    # add to containers
    msl_pis_min_lgm.append(  msl_pis_min  )
    delta_pis_max_lgm.append(delta_pis_max)

    # number of processed years
    nyrs = int(date_end.strftime('%Y')) - int(date_ini.strftime('%Y')) + 1

#SEGUIR COMO HISTOGRAMA

# logging message
indent = log_message('creating figure')
print(f'{indent}{output}')

# calculate weights
w_msl = []
w_delta = []

for i in range(len(msl_pis_min_lgm)):

    w_msl_i   = np.empty(  msl_pis_min_lgm[i].shape)
    w_delta_i = np.empty(delta_pis_max_lgm[i].shape)

    w_msl_i.fill(  1/len(  msl_pis_min_lgm[i]))
    w_delta_i.fill(1/len(delta_pis_max_lgm[i]))

    w_msl   += [w_msl_i]
    w_delta += [w_delta_i]

# colors of each simulation
colors = ['blue'   , 'cyan'   , 'limegreen'  ,  'orange', 'darkred']

# labels for each simulation (adding number of cyclones inside pis)
simids_str = [f"{simids[i]} (n = {len(msl_pis_min_lgm[i]):d})" for i in range(len(simids))]

# create figure
fig, axes = plt.subplots(2, 1, sharey=True, tight_layout=True)

# create msl & delta histogram
heights, bins, _  = axes[0].hist(msl_pis_min_lgm,   bins=bins_msl, density=False,
                                 histtype='bar', color=colors, label=simids_str,
                                 weights=w_msl)

heights, bins, _  = axes[1].hist(delta_pis_max_lgm, bins=bins_delta, density=False,
                                 histtype='bar', color=colors, label=simids_str,
                                 weights=w_delta)

# add grids
axes[0].grid(ls='--', color='grey', alpha=0.25)
axes[1].grid(ls='--', color='grey', alpha=0.25)

# set yticks
axes[0].set_yticklabels([f'{yt*100:.0f}%' for yt in axes[0].get_yticks()])
axes[1].set_yticklabels([f'{yt*100:.0f}%' for yt in axes[1].get_yticks()])

axes[0].set_xticks([np.mean(bins_msl[i:i+2]) for i in range(len(bins_msl)-1)], minor=True)
axes[1].set_xticks([np.mean(bins_delta[i:i+2]) for i in range(len(bins_delta)-1)], minor=True)

# set labels
axes[0].set_title(f"Sea level pressure (min. values)")
axes[1].set_title(f"SLP gradient over 1000km radius (max. values)")
axes[0].legend(prop={'size': 7})
axes[1].legend(prop={'size': 7})

# save / show plot
fig.savefig(f'{dirimg}/{output}', bbox_inches='tight')
#plt.show()





