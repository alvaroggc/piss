#!/home/cr2/agomez/.conda/envs/conda/bin/python
# -*- coding: utf-8 -*-
#
# ~/projects/piss/src/piss_hc_method_slp_minimums.py
#
# info  : identify cyclone centers in the southern hemisphere in piss
#         simulations (cesm2), based on Hanley & Caballero (2012).
# author: @alvaroggc


# standard libraries
import os
import gc
import re
import csv
import sys
import copy
import pickle
import datetime as dt

# 3rd party packages
import cf
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import multiprocess as mp
from pyproj import Geod
from shapely.geometry import Point, LineString
from shapely.geometry.polygon import Polygon

# local source
from _piss_lib import *


###########################
##### LOCAL FUNCTIONS #####
###########################


def slp_cyclonic_minimums(slp_k):
    '''
    info:
    parameters:
    returns:
    '''

    # get datetime string
    datestr = slp_k['time'].dt.strftime(fmt).item()

    # initiate timestep subcontainer
    cyclones = []

    # process each latitude gridpoint
    for i in range(1, len(y)-1):

        # process each longitude gridpoint
        for j in range(1, len(x)-1):

            # extract gridpoint latitude and altitude
            lat_ij  = LAT[ i, j]
            topo_ij = topo[i, j]

            # 9 points box (point in index 4 is the center)
            slp_k_box = slp_k[i-1:i+2, j-1:j+2].data.flatten()

            # separate points
            slp_k_boxcenter  = slp_k_box[4]
            slp_k_boxborders = np.delete(slp_k_box, 4)

            ### conditions to accept slp minimum
            # cond1: slp grid point is surrounded by higher values
            cond1 = ((slp_k_boxcenter < slp_k_boxborders).sum() == 8)

            # cond2: topo is lower than <max> value or higher than <max>
            #        but not in PIS (between 56°S and 38.5°S)
            cond2a = (topo_ij <= topo_max)
            cond2b = (lat_ij > -56) and (lat_ij < -38.5)

            cond2 = (cond2a or cond2b)

            # cond3: slp minimum south of 30°S
            cond3 = (lat_ij < -30)

            # check if center is minimum and [low or high but not in antartica]
            if cond1 and cond2 and cond3:

                # add x-y coordinates to subcontainer
                cyclones.append([np.float32(slp_k[i, j]['x'].item()),
                                 np.float32(slp_k[i, j]['y'].item()),
                                 np.float32(slp_k_boxcenter)])



    # process each point previously selected
    ipoint = 0
    while ipoint < len(cyclones):

        # separate point coordinates and values
        xp    = cyclones[ipoint][0]
        yp    = cyclones[ipoint][1]
        slp_p = cyclones[ipoint][2]

        # grids center in point
        XC = (slp_k['X'] - xp).data
        YC = (slp_k['Y'] - yp).data

        # mask that defines 1000 km area surrounding point
        mask = ((XC**2 + YC**2) < radius**2)

        # container for gradient values
        grad = []

        # process circle border
        for j in range(len(slp_k['x'])):

            # check if y-coordinate has, at least, one value to retrieve
            if (mask[:, j].sum() < 2):

                # skip y-coordinate
                continue

            # gradients in circle border
            grad.append((slp_k[mask[:, j], j][ 0] - slp_p).item())
            grad.append((slp_k[mask[:, j], j][-1] - slp_p).item())

        # mean gradient around center
        gradm = np.nanmean(grad)

        # next conditions to consider minimum as cyclone
        cond = (gradm >= grad_min)

        # check if center is minimum (if not, remove from container)
        if not cond:

            # remove point
            _ = cyclones.pop(ipoint)

        else:

            # add gradient to point container and border points of search radius
            cyclones[ipoint] += [np.float32(gradm)]

            # continue to next point
            ipoint += 1

    # create list of points for these slp minimums
    points    = [Point(p[0], p[1]) for p in cyclones]
    southpole = Point(0, 0)

    # minimum distance (km) between each contour line point
    eps = 180

    # calculate contour lines
    ipoint = 0
    while ipoint < len(cyclones):

        # create point object for reference
        slp_min_point = Point(cyclones[ipoint][0], cyclones[ipoint][1])

        # value of slp minimum
        slp_min = cyclones[ipoint][2]

        # create levels for contours
        levels = np.arange(slp_min, slp_k.max().item(), 2)

        # create figure just to calculate contour lines
        _fig, _ax = plt.subplots(1, 1, subplot_kw={'projection': proj})

        # plot field
        cont = slp_k.squeeze().plot.contour(ax=_ax,
                                            levels=levels,
                                            transform=transxy)

        # container for contour of slp minimum
        slp_cont = []

        # process each contour (skip first because it corresponds to center)
        for il, collection in enumerate(cont.collections[1:]):

            for path in collection.get_paths():

                # check if path not empty
                if not path.to_polygons():

                    # continue to next path
                    continue

                # extract vertices
                vert = path.vertices

                # coordinates of path
                xpath = vert[:, 0]
                ypath = vert[:, 1]

                # distances between each contour point coordinate
                difx = np.abs(np.diff(xpath))
                dify = np.abs(np.diff(ypath))

                # create contour polygon
                contourpoly = Polygon(   zip(xpath, ypath))
                contourline = LineString(zip(xpath, ypath))

                ### criteria to choose (and remove) contours
                # 1st: slp minimum inside contour
                cond1 = contourpoly.contains(slp_min_point)

                # 2nd: south pole NOT inside contour
                cond2 = (not contourpoly.contains(southpole))

                # 3er: close contour
                cond3 = contourline.is_closed

                # 4rd: number of slp minimums inside contour
                cond4 = [contourpoly.contains(pp) for pp in points]
                cond4 = (np.array(cond4).sum() <= 2)

                # 5th: contour not in border
                cond5 = ((difx < eps).all() and (dify < eps).all())

                # remove contour if all condicions are not fulfilled
                if (cond1 and cond2 and cond3 and cond4 and cond5):

                    # add to container
                    slp_cont += [(np.float32(levels[il+1]),
                                  xpath.astype(np.float32),
                                  ypath.astype(np.float32))]

        # check if center is minimum (if not, remove from container)
        if not slp_cont:

            # remove point
            _ = cyclones.pop(ipoint)

        else:

            # add gradient to point container and border points of search radius
            cyclones[ipoint] += [slp_cont[-1], datestr]

            # continue to next point
            ipoint += 1

        # close and delete uneccessary variables
        plt.close()
        del _fig
        del _ax


    # output
    return cyclones


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
    ra_fmt    = re.compile(r'^ra_(6|24)h$')

    # retrieve variables
    simid = [arg for arg in args if simid_fmt.match(str(arg))]   # sim. ID
    simid = [arg for arg in args if ra_fmt.match(str(arg))]

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
dirout  = f'{homedir}/projects/piss/data'       # data output

# indexer to choose southern hemisphere (from 0° to the south)
shidx = {'lat': slice(-90, 0)}

# parameters needed in method
topo_max = 1500     # slp minimum points over this altitude are eliminated [m]
radius   = 1000     # gradient search radius [km]
grad_min = 10       # minimum slp gradient required for cyclone [hPa]

# temporal range
date_ini = '0001-01-01 00:00:00' if ('ra' not in simid) else '1990-01-01 00:00:00'
date_end = '0036-01-01 00:00:00' if ('ra' not in simid) else '2001-01-01 00:00:00'

# date string format
fmt = '%Y-%m-%d %H:%M'


###################
##### RUNNING #####
###################


# load simulation datasets
slp  = load_simulation_variable(dirsim, simid, 'PSL')
topo = load_simulation_variable(dirsim, simid, 'PHIS')

# extract temporal range (only for slp)
slp = slp.sel({'time': slice(date_ini, date_end)})

# range of years
yri =  slp['time.year'][ 0].item()
yrf =  slp['time.year'][-1].item()

# output file with cyclones information
fout = f'cyclones_{simid}_{yri:04d}_{yrf:04d}.pkl'

# extract southern hemisphere
slp  = slp.sel( shidx)
topo = topo.sel(shidx)

# adjust slp units
slp = slp.where(False, slp / 100)
slp.attrs['units'] = 'hPa'

# adjust topography data
topo = topo.where(False, topo / 9.8)  # geop. to geop. height
topo = topo.isel({'time': [0]})       # leave first timestep

# if grids are different, interpolate to topography grid (only happens with ra)
if ('ra' in simid):

    slp = slp.interp({'lat': topo['lat'],
                      'lon': topo['lon']}, method='linear')

# load data
slp  = slp.load()
topo = topo.load()

# resample data (only for ra dataset)
if (simid == 'ra_24h'):

    # from 6h resolution to 24h res.
    slp = slp.isel({'time': slice(None, None, 4)})

# convert coordinates to lambert projection
slp  = convert_to_lambert(slp)
topo = convert_to_lambert(topo).squeeze()

# only retain values of topo (not necessary whole xarray container)
topo = topo.data

# separate coordinates
x = slp['x'].data
y = slp['y'].data

# create lat grid based on x-y coordinates
X, Y = np.meshgrid(x, y, indexing='ij')
LAT  = convert_point_to_latlon((X.flatten(), Y.flatten()))[1]
LAT  = LAT.reshape(Y.shape)

# delete unnecesary variable (to free memory)
del X
del Y

# arguments for slp minimum selection process function
args = xarray_time_iterator(slp)

# create cyclones container
cyclones = {}

# slp minimum identifier
sid = 0

# logging message
indent = log_message('starting slp minimum identification')

# create thread pool
with mp.Pool(processes=25) as pool:

    # compute processing
    results = pool.imap(slp_cyclonic_minimums, args)

    # fill container
    for k, cyclones_k in enumerate(results):

        # key to dictionary (date)

        datestr = slp.isel({'time': k})['time'].dt.strftime(fmt).item()

        # add to dictionary
        cyclones[datestr] = cyclones_k

        # add cyclone identifier to points
        for point in cyclones[datestr]:

            point.append(sid)
            sid +=1

        # logging message
        indent = log_message(f'slp centers found in {datestr}')
        print(f'{indent}{len(cyclones_k)} (second selection) ')

# save dictionary with slp minimum centers
with open(f'{dirout}/{fout}', 'wb') as f:

    # write to file
    pickle.dump(cyclones, f, protocol=pickle.HIGHEST_PROTOCOL)


