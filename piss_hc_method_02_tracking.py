#!/home/cr2/agomez/.conda/envs/conda/bin/python
# -*- coding: utf-8 -*-
#
# ~/projects/piss/src/piss_hc_method_02_tracking.py
#
# info  : track cyclones in the souther hemisphere in piss simulations (cesm2), based on
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


def contour_search(da, point):
    '''
    info:
    parameters:
    returns:
    '''

    # separate coordinates of field
    lat   = da['lat']
    lon   = da['lon']
    dates = da['time']

    # separate coordinates of point
    latp  = point[0]
    lonp  = point[1]
    datep = cf.dt(point[-1], calendar='noleap')

    # find indexes of point inside coordinates
    ilat, = np.where(np.abs((lat - latp)) == np.min(np.abs((lat - latp))))[0]
    jlon, = np.where(np.abs((lon - lonp)) == np.min(np.abs((lon - lonp))))[0]
    kt  , = np.where(np.abs((dates - datep)) == np.min(np.abs((dates - datep))))[0]

    # choose date
    dak = da.isel({'time': kt}).squeeze()

    # # process each contour
    # while ((ilat < len(lat)) and (ilon < len(lon))): # <-- option 1, i think it won't work
    #
    #     # for now, do nothing
    #     None


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
    D_fmt     = re.compile(r'^D\d')             # search distance

    # retrieve variables
    simid = [arg for arg in args if simid_fmt.match(str(arg))]   # sim. ID
    D     = [arg for arg in args if D_fmt.match(    str(arg))]   # search distance

    # check arguments
    simid = 'lgm_100' if not simid else simid[0]
    D     = 1000      if not D     else int(D[0][1:])

    # output
    return (simid, D)


############################
##### LOCAL PARAMETERS #####
############################


# get variables from input arguments
simid, D = get_variables(sys.argv)

# directories
homedir = os.path.expanduser('~')               # home directory
dirin   = f'/mnt/cirrus/results/friquelme'      # cesm simulations
dirout  = f'{homedir}/projects/piss/data'       # data output
dirimg  = f'{homedir}/projects/piss/img'        # output for figures

# filenames for stored results
fin  = f'cyclones_{simid}_v3_0001_0001.pkl'

# variable to process
key = 'PSL'

# indexer to choose southern hemisphere (from -30° to the south)
shidx = {'lat': slice(-90, -30)}

# parameters needed in method
D0 = 1200   # threshold distance to track cyclones [km]
# D0 = D    # threshold distance to track cyclones [first timesteps]


###################
##### RUNNING #####
###################


# execution time
time_ini = dt.datetime.today()

# open cyclones file
with open(f'{dirout}/{fin}', 'rb') as f:

    # read file
    cyclones = pickle.load(f)

# temporal range
date_ini = [*cyclones.keys()][ 0]
date_end = [*cyclones.keys()][-1]

# load simulation datasets
slp = load_simulation(dirin, simid, ['PSL'])['PSL']

# extract temporal range (only for slp)
slp  = slp.sel( {'time': slice(date_ini, date_end)})

# range of years
yri = slp['time.year'][ 0].item()
yrf = slp['time.year'][-1].item()

# output file with cyclones tracks information
fout = f'tracks_{simid}_v3_{yri:04d}_{yrf:04d}.pkl'

# extract southern hemisphere
slp  = slp.sel(shidx).load()

# adjust slp units
slp = slp.where(False, slp / 100)
slp.attrs['units'] = 'hPa'

# convert coordinates to lambert projection [latlon -> xy]
slp = convert_to_lambert(slp)

# track cyclone centers
# * point = [y[0], x[1], slp[2], grad_slp[3], contours[4], date[5], sid[6]]

# cyclone code and id
cn  = 0             # cyclone number
cid = f'c{cn:05d}'  # cyclone identification code

# logging message
indent = log_message('assembling tracks')

# tracks container
tracks = {}

# process each date
for it in range(len(slp['time'])-2):

    # date identifiers for initial step of track
    datestr0 = slp['time'][it+0].dt.strftime('%Y-%m-%d').item()
    datestr1 = slp['time'][it+1].dt.strftime('%Y-%m-%d').item()
    datestr2 = slp['time'][it+2].dt.strftime('%Y-%m-%d').item()

    # process each point of timestep
    ip0 = 0
    while ip0 < len(cyclones[datestr0]):

        # extract point
        p0 = cyclones[datestr0][ip0]

        # create containers for chosen points
        trackpoints = []
        idx         = []

        # minimum distance neccesary to choose points for track
        dist_min = D

        # process points of next timestep
        for ip1, p1 in enumerate(cyclones[datestr1]):

            # distance between points (different timesteps)
            d01  = distance_between_points(p0, p1)
            east = east_or_not(p0, p1)

            # check if distance is less than initial threshold
            if (d01 < D0) and (east):

                # guess next point
                p2_guess = guess_point(p0, p1)

                # process points of third timestep
                for ip2, p2 in enumerate(cyclones[datestr2]):

                    # distance between point and guess (same timestep)
                    dist_guess = distance_between_points(p2, p2_guess)

                    # check if point is east to previous point
                    east = east_or_not(p1, p2)

                    # check if distance is less than threshold
                    if (dist_guess < dist_min) and (east):

                        # update minimum distance
                        dist_min = dist_guess

                        # replace points in containers
                        trackpoints = [ p0,  p1,  p2]
                        idx         = [ip0, ip1, ip2]

        # check if track was initialized
        if not trackpoints:

            # if not initialized, continue to next point
            ip0 += 1
            continue

        # remove points from original container
        _ = cyclones[datestr0].pop(idx[0])
        _ = cyclones[datestr1].pop(idx[1])
        _ = cyclones[datestr2].pop(idx[2])

        # continue adding points to track
        for jt in range(it+2, len(slp['time'])-1):

            # date identifier
            datestr_next = slp['time'][jt+1].dt.strftime('%Y-%m-%d').item()

            # extraction of last points in track
            plast = trackpoints[-1]
            pprev = trackpoints[-2]

            # guess for next point in track
            pnext_guess = guess_point(pprev, plast)

            # minimum distance neccesary to choose points for track
            dist_min = D

            # new flag to check if continuation of track was successful
            success = False

            # process each point in next timestep
            for ipnext, pnext in enumerate(cyclones[datestr_next]):

                # distance between point and guess (same timestep)
                dist_guess = distance_between_points(pnext, pnext_guess)

                # check if point is east to last trackpoint
                east = east_or_not(plast, pnext)

                # check if distance is less than threshold
                if (dist_guess < dist_min) and (east):

                    # update minimum distance
                    dist_min = dist_guess

                    # replace point in containers
                    newpoint = pnext
                    idx      = ipnext

                    # update continuation flag
                    success = True

            # check if new point needs to be added to track
            if not success:

                # add final track to tracks container
                tracks[cid] = trackpoints

                # logging message
                print(f'{indent}{datestr0}: {cid} ({len(trackpoints):02d})')

                # update cyclone code and id
                cn += 1
                cid = f'c{cn:05d}'

                # as track ended, continue to next point
                break

            # add new point to track
            trackpoints.append(newpoint)

            # remove point from original container
            _ = cyclones[datestr_next].pop(idx)

# save dictionary with cyclone tracks in file
with open(f'{dirout}/{fout}', 'wb') as f:

    # write file
    pickle.dump(tracks, f, protocol=pickle.HIGHEST_PROTOCOL)

# final execution time
time_end = dt.datetime.today()

# logging message
indent = log_message('excecution time')
print(f'{indent}{(time_end - time_ini) / dt.timedelta(hours=1)} hours')


























