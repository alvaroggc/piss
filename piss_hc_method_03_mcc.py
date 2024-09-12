#!/home/cr2/agomez/.conda/envs/conda/bin/python
# -*- coding: utf-8 -*-
#
# ~/projects/piss/src/piss_hc_method_03_mcc.py
#
# info  : track surgery in piss simulations (cesm2), based on
#         Hanley & Caballero (2012).
# author: @alvaroggc


# standard libraries
import os
import re
import csv
import sys
import copy
import pickle
import datetime as dt

# 3rd party packages
import cf
import cartopy.crs as ccrs
from pyproj import Geod
from shapely.geometry import Point, LineString
from shapely.geometry.polygon import Polygon

# local source
from piss_lib import *


###########################
##### LOCAL FUNCTIONS #####
###########################


def load_data(dirin, simid, dtype='cyclones', ctype = 'dict'):
    '''
    info:
    parameters:
    returns:
    '''

    # load cyclone tracks of simulation
    with open(f'{dirin}/{dtype}_{simid}_v3_0001_0001.pkl', 'rb') as f:

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

    # retrieve variables
    simid = [arg for arg in args if simid_fmt.match(str(arg))]   # sim. ID

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
dirimg  = f'{homedir}/projects/piss/img'        # output for figures

# indexer to choose southern hemisphere (from -30° to the south)
shidx = {'lat': slice(-90, -30)}

# figure parameters
size = (7.5, 6)


###################
##### RUNNING #####
###################

# execution time
time_ini = dt.datetime.today()

# open files
tracks   = load_data(dirdata, simid, dtype='tracks',   ctype = 'list')
cyclones = load_data(dirdata, simid, dtype='cyclones', ctype = 'list')

# get temporal range
dates = [*np.unique([cyclone[5] for cyclone in cyclones])]

# range of years
yri = int(dates[ 0].split('-')[0])
yrf = int(dates[-1].split('-')[0])

# output file with cyclones tracks (mcc) information
fout = f'tracks_mcc_{simid}_v3_{yri:04d}_{yrf:04d}.pkl'

# extract list of cyclone and track IDs in tracks container
tid = [track[-1] for track in tracks]
cid = [f'{track[-2]:d}' for track in tracks]

# mcc identifier
mccid = 0

# logging message
indent = log_message('searching for mcc')

# process each date
for date in dates:

    # extract points in date
    cyclones_date = [cyclone for cyclone in cyclones if cyclone[5] == date]

    # extract all cyclone points
    points = [Point(p[1], p[0]) for p in cyclones_date]

    # process each point
    for p in cyclones_date:

        # code of cyclone point
        ccode = str(p[-1])

        # logging message
        print(f'{indent}{date}: {ccode}')

        # flags that indicate if points in any track
        intracks = (ccode in cid)

        # check if cyclone point is in any track
        if not intracks:

            # skip to next point
            continue

        # get code of track that point belongs to
        tcode = tid[ccode == cid]

        # value of slp minimum
        slpmin = p[2]

        # create point object
        point = Point(p[1], p[0])

        # copy of all data from points without the one being processed
        cyclones_datex = cyclones_date.copy()
        cyclones_datex.remove(p)

        # copy of all coordinate points without the one being processed
        pointsx = points.copy()
        pointsx.remove(point)

        # get last contour of point
        slpc = p[4][0]
        xc   = p[4][1]
        yc   = p[4][2]
        cc   = Polygon(zip(xc, yc))

        # check if any point in this timestep is inside contour
        inside = [cc.contains(pp) for pp in pointsx]

        # get indexes with inside points
        idx = [*np.where(inside)[0]]

        # case 1: one other slp minimum inside last contour
        if (len(idx) == 1):

            # code of cyclone point
            ccode_1 = cyclones_datex[idx[0]][-1]

            # flags that indicate if points in any track
            intracks_1 = (ccode_1 in cid)

            # values of slp minimums
            slpmin_1 = cyclones_datex[idx[0]][2]

            # get value of outter contour
            slpc_1 = cyclones_datex[idx[0]][4][-1][0]

            # if point not in any track
            if not intracks_1:

                # continue to next point
                continue

            # get extreme slp values to compute number of contours
            slpmin_min = np.min([slpmin, slpmin_1])
            slpmin_max = np.max([slpmin, slpmin_1])
            slpc_min   = np.min([slpc,   slpc_1])
            slpc_max   = np.max([slpc,   slpc_1])

            # ratio of shared contours
            rc = 0.5

        # case 2: two other slp minimum inside last contour
        elif (len(idx) == 2):

            # codes of cyclone points
            ccode_1 = cyclones_datex[idx[0]][-1]
            ccode_2 = cyclones_datex[idx[1]][-1]

            # flags that indicate if points in any track
            intracks_1 = (ccode_1 in cid)
            intracks_2 = (ccode_2 in cid)

            # values of slp minimums
            slpmin_1 = cyclones_datex[idx[0]][2]
            slpmin_2 = cyclones_datex[idx[1]][2]

            # get values of outter contours
            slpc_1 = cyclones_datex[idx[0]][4][-1][0]
            slpc_2 = cyclones_datex[idx[1]][4][-1][0]

            # get extreme slp values to compute number of contours
            if intracks_1 and not intracks_2:

                slpmin_min = np.min([slpmin, slpmin_1])
                slpmin_max = np.max([slpmin, slpmin_1])
                slpc_min   = np.min([slpc,   slpc_1])
                slpc_max   = np.max([slpc,   slpc_1])

            elif not intracks_1 and intracks_2:

                slpmin_min = np.min([slpmin, slpmin_2])
                slpmin_max = np.max([slpmin, slpmin_2])
                slpc_min   = np.min([slpc,   slpc_2])
                slpc_max   = np.max([slpc,   slpc_2])

            elif intracks_1 and intracks_2:

                slpmin_min = np.min([slpmin, slpmin_1, slpmin_2])
                slpmin_max = np.max([slpmin, slpmin_1, slpmin_2])
                slpc_min   = np.min([slpc,   slpc_1, slpc_2])
                slpc_max   = np.max([slpc,   slpc_1, slpc_2])

            else:

                # continue to next point
                continue

            # ratio of shared contours
            rc = 0.7

        # case 3: three or more slp minimums inside last contour
        elif (len(idx) > 2):

            # logging message
            input(f'too many points inside contour: FIX')

        # case 4: no points inside contour (excluding center point)
        else:

            # continue to next point
            continue

        # number of share contours
        # nshared = len(np.arange(slpmin_max, slpc_max, 2))
        nshared = len(np.arange(slpc_max, slpmin_max, -2))

        # number of total contours
        # ntotal = len(np.arange(slpmin_min, slpc_max, 2))
        ntotal = len(np.arange(slpc_max, slpmin_min, -2))

        # mcc ratio
        ratio = nshared / ntotal

        # check if mcc (two points)
        if (ratio > rc) and (len(idx) == 1):

            # update cyclone point ID
            None
            input('two points, stop')

        # check if mcc (three points)
        elif (ratio > rc) and (len(idx) == 2):

            # update cyclone point ID
            None
            input('three points, stop')

# track cyclone centers
# * point = [y[0], x[1], slp[2], grad_slp[3], cont[4], date[5], sid[6]]


# final execution time
time_end = dt.datetime.today()

# logging message
indent = log_message('excecution time')
print(f'{indent}{(time_end - time_ini) / dt.timedelta(hours=1)} hours')


























