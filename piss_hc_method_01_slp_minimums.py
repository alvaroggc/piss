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
import cartopy.crs as ccrs
import multiprocess as mp
from pyproj import Geod
from shapely.geometry import Point, LineString
from shapely.geometry.polygon import Polygon

# local source
from piss_lib import *


###########################
##### LOCAL FUNCTIONS #####
###########################




def east_or_not(p1, p2, D=500):
    '''
    info:
    parameters:
    returns:
    '''

    # convert to lat-lon coordinate system
    lat1, lon1 = convert_point_to_latlon(p1)
    lat2, lon2 = convert_point_to_latlon(p2)

    # longitude alternative
    lon1aux = (lon1 - 360) if (lon1 > 180) else lon1
    lon2aux = (lon2 - 360) if (lon2 > 180) else lon2

    # check if p2 is east of p1
    east = False if ((lon2 < lon1) and (lon2aux < lon1aux)) else True

    # if not moving east, at least check if p2 doesn't move much from p1
    if not east:

        # distance between points
        dist = distance_between_points(p1, p2)

        # check if distance is low
        if (dist < D):

            # update flag to indicate that p2 is a valid point
            east = True

    # output
    return east


def guess_point(p1, p2):
    '''
    info:
    parameters:
    returns:
    '''

    # convert to lat-lon coordinate system
    lat1, lon1 = convert_point_to_latlon(p1)
    lat2, lon2 = convert_point_to_latlon(p2)

    # if point to close to 0°lon, change range of longitude
    if (lon1 > 320) or (lon2 > 320):

        # shift coordinates
        lon1 = (lon1 - 360) if (lon1 > 180) else lon1
        lon2 = (lon2 - 360) if (lon2 > 180) else lon2

    # guess next point
    lat3 = lat2 + 0.75 * (lat2 - lat1)
    lon3 = lon2 + 0.75 * (lon2 - lon1)

    # convert to x-y coordinate
    p_guess = convert_point_to_xy([lat3, lon3])

    # output
    return p_guess


def slp_cyclonic_minimums(slp_k):
    '''
    info:
    parameters:
    returns:
    '''

    # get datetime string
    datestr = slp_k['time'].dt.strftime('%Y-%m-%d').item()

    # initiate timestep subcontainer
    cyclones = []

    # process each latitude gridpoint
    for i in range(1, len(slp_k['y'])-1):

        # process each longitude gridpoint
        for j in range(1, len(slp_k['x'])-1):

            # 9 points box (point in index 4 is the center)
            slp_k_box  = slp_k[ i-1:i+2, j-1:j+2].data.flatten()
            # topo_xy_box = topo_xy[i-1:i+2, j-1:j+2].data.flatten()

            # separate points
            slp_k_boxcenter  = slp_k_box[4]
            slp_k_boxborders = np.delete(slp_k_box, 4)
            # topo_xy_boxcenter = topo_xy_box[4]

            # initial conditions to consider minimum as cyclone
            cond1 = ((slp_k_boxcenter  <  slp_k_boxborders).sum() == 8)
            # cond2 = ( topo_xy_boxcenter <= topo_max)

            # check if center is minimum
            # if cond1 and cond2:
            if cond1:   # test to see topography influence

                # add x-y coordinates to subcontainer
                cyclones.append([np.float32(slp_k[i, j]['y'].item()),
                                 np.float32(slp_k[i, j]['x'].item()),
                                 np.float32(slp_k_boxcenter)])

    # process each point previously selected
    ipoint = 0
    while ipoint < len(cyclones):

        # separate point coordinates and values
        yp    = cyclones[ipoint][0]
        xp    = cyclones[ipoint][1]
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
            if (mask[j, :].sum() < 2):

                # skip y-coordinate
                continue

            # gradients in circle border
            grad.append((slp_k[j, mask[j, :]][ 0] - slp_p).item())
            grad.append((slp_k[j, mask[j, :]][-1] - slp_p).item())

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
    points    = [Point(p[1], p[0]) for p in cyclones]
    southpole = Point(0, 0)

    # minimum distance (km) between each contour line point
    eps = 180

    # calculate contour lines
    ipoint = 0
    while ipoint < len(cyclones):

        # create point object for reference
        slp_min_point = Point(cyclones[ipoint][1], cyclones[ipoint][0])

        # value of slp minimum
        slp_min = cyclones[ipoint][2]

        # create levels for contours
        levels = np.arange(slp_min, slp_k.max().item(), 2)

        # create figure just to calculate contour lines
        _fig, _ax = plt.subplots(1, 1)

        # plot field
        cont = slp_k.squeeze().plot.contour(ax=_ax, levels=levels)

        # container for contour of slp minimum
        slp_cont = []

        # process each contour
        for il, collection in enumerate(cont.collections):

            for path in collection.get_paths():

                # check if path not empty
                if not path.to_polygons():

                    # continue to next path
                    continue

                # coordinates of path
                xpath = path.to_polygons()[0][:, 0]
                ypath = path.to_polygons()[0][:, 1]

                # distances between each contour point coordinate
                difx = np.abs(np.diff(xpath))
                dify = np.abs(np.diff(ypath))

                # create contour polygon
                contourpoly = Polygon(   zip(xpath, ypath))
                contourline = LineString(zip(xpath, ypath))

                # criteria to choose (and remove) contours
                # 1st: slp minimum inside contour
                cond1 = contourpoly.contains(slp_min_point)

                # 2nd: pole NOT inside contour
                cond2 = (not contourpoly.contains(southpole))

                # 3er: close contour
                cond3 = contourline.is_closed

                # 4rd: number of slp minimums inside contour
                cond4 = [contourpoly.contains(pp) for pp in points]
                cond4 = (np.array(cond4).sum() <= 3)

                # 5th: contour not in border
                cond5 = ((difx < eps).all() and (dify < eps).all())

                # remove contour if all condicions are not fulfilled
                if (cond1 and cond2 and cond3 and cond4 and cond5):

                    # add to container
                    slp_cont += [(np.float32(levels[il]),
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
dirout  = f'{homedir}/projects/piss/data'       # data output

# indexer to choose southern hemisphere (from -30° to the south)
shidx = {'lat': slice(-90, -30)}

# parameters needed in method
topo_max = 1000     # slp minimum points over this altitude are eliminated [m]
radius   = 1000     # gradient search radius [km]
grad_min = 10       # minimum slp gradient required for cyclone [hPa]

# temporal range
date_ini = '0001-01-01 00:00:00'
date_end = '0001-12-01 00:00:00'


###################
##### RUNNING #####
###################


# load simulation datasets
ds = load_simulation(dirsim, simid, ['PSL', 'PHIS'])

# separate variables
slp  = ds['PSL']
topo = ds['PHIS']

# remove dataset variable (to clean memory usage)
del ds

# extract temporal range (only for slp)
slp  = slp.sel( {'time': slice(date_ini, date_end)})

# range of years
yri =  slp['time.year'][ 0].item()
yrf =  slp['time.year'][-1].item()

# output file with cyclones information
fout = f'cyclones_{simid}_v3_{yri:04d}_{yrf:04d}.pkl'

# extract southern hemisphere
slp = slp.sel(shidx).load()
topo = topo.sel(shidx).load()

# adjust slp units
slp = slp.where(False, slp / 100)
slp.attrs['units'] = 'hPa'

# adjust topography data
topo = topo.where(False, topo / 9.8)            # geop. to geop. height
topo = topo.sel({'time': topo['time'][[0]]})    # leave first timestep
topo.attrs['units'] = 'm'

# convert coordinates to lambert projection
slp  = convert_to_lambert(slp)
topo = convert_to_lambert(topo).squeeze()

# arguments for slp minimum selection process function
args = xarray_time_iterator(slp)

# create cyclones container
cyclones = {}

# slp minimum identifier
sid = 0

# create thread pool
with mp.Pool(processes=25) as pool:

    # compute processing
    results = pool.imap(slp_cyclonic_minimums, args)

    # fill container
    for k, cyclones_k in enumerate(results):

        # key to dictionary (date)
        datestr = slp.isel({'time': k})['time'].dt.strftime('%Y-%m-%d').item()
        dateid  = slp.isel({'time': k})['time'].dt.strftime(' %Y%m%d ').item()

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


