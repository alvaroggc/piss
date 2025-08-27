#!/home/cr2/agomez/.conda/envs/conda/bin/python
# -*- coding: utf-8 -*-
#
# ~/projects/piss/src/piss_pproc_sensibility.py.py
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


############################
##### LOCAL PARAMETERS #####
############################


# directories
homedir = os.path.expanduser('~')               # home directory
projdir = f'{homedir}/projects/piss'

dirsim  = f'/mnt/cirrus/results/friquelme'      # cesm simulations
dirdata = f'{projdir}/data'      # cesm simulations
dirimg  = f'{projdir}/img/extra'        # output for figures

# output image filename
output = f'piss_sensibility_pis.png'

# list of all simulation names
simids = ['lgm_050', 'lgm_100', 'lgm_150', 'lgm_200', 'lgm_300']

# indexer to choose southern hemisphere (from 20°S to the south)
shidx = {'lat': slice(-90, 0)}

# figure parameters
size = (7.5, 3.5)

# text box parameters
bbox = {'boxstyle' : 'round',
        'facecolor': 'white',
        'edgecolor': 'black',
        'alpha'    : 1.0}


###################
##### RUNNING #####
###################


# path to ice-mask
fice = f'{dirdata}/pis_mask.nc'

# load ice-mask
ice = xr.open_dataset(fice)['ice']

# # fix longitude to be cyclic
# lon = ice['lon'].data
# lon[-1] = 360
# ice['lon'] = ice['lon'].where(False, lon)

# extract southern hemisphere
ice = ice.sel(shidx)

# convert coordinate system (from lat-lon to x-y)
ice = convert_to_lambert(ice, method='nearest')

# convert coordinates units (from km to m) (only 1D)
ice['x'] = ice['x'].where(False, ice['x'] * 1000)
ice['y'] = ice['y'].where(False, ice['y'] * 1000)

# count['x'] = count['x'].where(False, count['x'] * 1000)
# count['y'] = count['y'].where(False, count['y'] * 1000)

ice['X'] = ice['X'].where(False, ice['X'] * 1000)
ice['Y'] = ice['Y'].where(False, ice['Y'] * 1000)

# count['X'] = count['X'].where(False, count['X'] * 1000)
# count['Y'] = count['Y'].where(False, count['Y'] * 1000)

# expand pismask
ice = expand_pismask(ice)

# create container with average ocurrence information of cylones inside pis
avg_pis    = []
delta_topo = []

# update output directory and create if it doesn't exists
os.makedirs(f'{dirimg}') if not os.path.isdir(f'{dirimg}') else None

# logging message
indent = log_message('creating sensibility diagram')

# process each simulation
for simid in simids:

    # path to storm track ocurrences
    fcount = f'{dirdata}/cyclonic_ocurrences_{simid.lower()}.nc'

    # logging message
    print(f"{indent}{fcount.split('/')[-1]}")

    # load cyclonic frequency ocurrence
    count = xr.open_dataset(fcount)['count']

    # # regrid ice data
    # ice = ice.interp({'lat': count['lat'], 'lon': count['lon']}, method='linear')
    # ice = ice.where(ice == 0, 1)

    # topography factor
    delta_topo.append(f"{int(simid.split('_')[-1])}%")

    # average ocurrence inside pis
    avg_pis.append(count.where(ice.astype(bool)).mean(skipna=True).item())

# convert to numeric values
avg_pis = np.array(avg_pis)
delta_topo_num = np.array([int(val[:-1]) for val in delta_topo])

##########
# FIGURE #
##########


# create figure
fig, ax = plt.subplots(1, 1, figsize=size)#, subplot_kw={'projection': proj})

# show response of ocurrence vs topography factor
ax.plot(delta_topo_num, avg_pis, ls='none', marker='o', mfc='red',  mec='none', alpha=0.5)
ax.plot(delta_topo_num, avg_pis, ls='none', marker='o', mfc='none', mec='black')
ax.plot(delta_topo_num, avg_pis, ls='--', color='black', zorder=0, alpha=0.25)

# define labels [see Hanley & Caballero (2012),
# figure 7a to understand 7.5 degree circle]
ax.set_xlabel('Topography increase of PIS')
ax.set_xticks(delta_topo_num)
ax.set_xticklabels(delta_topo)
ax.set_ylabel('N° of cyclone tracks')
ax.set_title(f'PIS average cyclone tracks per year per 7.5 degree circle')

# set grid
ax.grid(color='grey', ls='--', alpha=0.25)
# set coordinate limits
# ax.set_extent([250, 350, -60, -30])#, crs=trans)

# save / show plot
fig.savefig(f'{dirimg}/{output}', bbox_inches='tight')
#plt.show()

# save pis mask
# da.to_netcdf(f'{dirin}/pis_mask.nc')
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
