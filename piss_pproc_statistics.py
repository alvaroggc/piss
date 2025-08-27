#!/home/alvaro/.conda/envs/conda/bin/python
# -*- coding: utf-8 -*-
#
# ~/projects/piss/src/piss_pproc_statistics.py
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

dirsim  = f'/mnt/cirrus/results/friquelme'      # cesm simulations
dirdata = f'{projdir}/data'      # cesm simulations

# list of all simulation names
simids = ['lgm_050', 'lgm_100', 'lgm_150', 'lgm_200', 'lgm_300',
          'ra_6h', 'ra_24h']


###################
##### RUNNING #####
###################


# create container with average ocurrence information of cylones inside pis
avg_pis    = []
delta_topo = []

# logging message
indent = log_message('counting elements')

# process each simulation
for simid in simids:

    # open preview results
    cyclones = load_results(dirdata, simid, dtype='cyclones', ctype='list')
    tracks   = load_results(dirdata, simid, dtype='tracks'  , ctype='list')

    # count number of elements
    ncyclones = len(cyclones)
    ntracks   = len(tracks)

    # logging message
    print(f"{indent}{simid}: {(ntracks / ncyclones * 100):.2f}%")


