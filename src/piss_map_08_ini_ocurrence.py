#!/home/cr2/agomez/.conda/envs/conda/bin/python
# -*- coding: utf-8 -*-
#
# ~/projects/piss/src/piss_map_08_ini_ocurrence.py
#
# info  : show initial ocurrence of cyclonic activity over southern hemisphere
#         in piss simulations (cesm2), based on Hanley & Caballero (2012).
# usage : ./piss_map_08_ini_ocurrence.py
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


def count_tracks(ij):
    '''
    info:
    parameters:
    returns:
    '''

    # extract coordinates of gridpoint
    xref = x[ij[0]]
    yref = y[ij[1]]

    # create mask with distances from gridpoint
    dist = distance_between_points((xref, yref), (xp, yp))

    # mask with points inside search radius
    mask = (dist < D)

    # list with unique cyclone tracks ids
    valid_tracks = np.unique(tids[mask])

    # add to counter of any of the points in track is inside search radius
    count_ij = len(valid_tracks)

    # logging message
    print(f'{indent}({xref:5.1f}, {yref:5.1f}) {count_ij:.0f}')

    # output
    return count_ij


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
    ra_fmt    = re.compile(r'^ra_(6|24)h$')
    pis_sa_fmt = re.compile(r'^(pis|sa)$')

    # retrieve variables
    simid = [arg for arg in args if simid_fmt.match(str(arg))]   # sim. ID
    raid  = [arg for arg in args if ra_fmt.match(str(arg))]

    simid = raid if raid else simid

    pis_sa = [f'_{arg}' for arg in args if pis_sa_fmt.match(str(arg))]

    nocoast = '_nocoast' if 'nocoast' in args else ''
    anom    = 'a'        if 'anom'    in args else ''
    calc    = False      if 'nocalc'  in args else True

    # check arguments
    simid = 'lgm_100' if not  simid else  simid[0]
    pis_sa =       '' if not pis_sa else pis_sa[0]

    # output
    return (simid, pis_sa, nocoast, anom, calc)


############################
##### LOCAL PARAMETERS #####
############################


# get variables from input arguments
simid, pis_sa, nocoast, anom, calc = get_variables(sys.argv)

# directories
homedir = os.path.expanduser('~')               # home directory
projdir = f'{homedir}/projects/piss'

dirsim  = f'/mnt/cirrus/results/friquelme'      # cesm simulations
dirdata = f'{projdir}/data'                     # data output
dirimg  = f'{projdir}/img/ocurrences_ini/{nocoast[1:]}'    # output for figures

# start date (first 5 years are transitional)
calendar   = 'gregorian' if ('ra' in simid) else 'noleap'
date_start = cf.dt('0005-01-01 00:00:00', calendar=calendar)

# indexer to choose southern hemisphere (from -30° to the south)
shidx = {'lat': slice(-90, -30)}

# parameters for method
D = 840000     # 840 km

# smooth radius
r = 2

# figure parameters
size = (7.5, 6) if (not 'pis' in pis_sa) else (15, 12)


###################
##### RUNNING #####
###################


if calc:

    # load simulation datasets
    ref = xr.open_dataset(f'{dirdata}/pis_mask.nc')['ice']

    # only leave southern hemisphere
    ref = ref.sel({'lat': slice(-90, 0)}).load()

    # convert coordinate system (from lat-lon to x-y)
    ref = convert_to_lambert(ref)

    # convert coordinates units (from km to m)
    ref['x'] = ref['x'].where(False, ref['x'] * 1000)
    ref['y'] = ref['y'].where(False, ref['y'] * 1000)

    ref['X'] = ref['X'].where(False, ref['X'] * 1000)
    ref['Y'] = ref['Y'].where(False, ref['Y'] * 1000)

    # separate spatial coordinates (1D)
    x = ref['x'].data
    y = ref['y'].data

    # delete ref variable (to free memory)
    del ref

    # load cyclone tracks of simulation
    tracks = load_results(dirdata, simid, dtype='tracks', ctype='dict')

    # get tracks keys list
    tids = [*tracks.keys()]

    # extract points of first step of each track
    xp = [tracks[tid][0][0] for tid in tids]
    yp = [tracks[tid][0][1] for tid in tids]
    tp = [cf.dt(tracks[tid][0][5], calendar=calendar) for tid in tids]

    # convert to arrays
    xp   = np.array(xp)
    yp   = np.array(yp)
    tp   = np.array(tp)
    tids = np.array(tids)

    # convert units (from km to meters)
    xp *= 1000
    yp *= 1000

    # remove first 5 transitional years
    mask5years = (tp >= date_start)

    xp   =   xp[mask5years]
    yp   =   yp[mask5years]
    tp   =   tp[mask5years]
    tids = tids[mask5years]

    # get temporal range
    date_ini = tp[ 0]
    date_end = tp[-1]

    # # separate tracks components
    # xp = np.array([p[ 0] for p in tracks])                        # y-coordinate
    # yp = np.array([p[ 1] for p in tracks])                        # x-coordinate
    #
    # dt_str_p = [p[5] for p in tracks]
    # dt_str_p = dt_str_p.replace('H', '') if 'H' in dt_str_p else dt_str_p
    # dt_str_p = dt_str_p.replace('M', '') if 'M' in dt_str_p else dt_str_p
    # tp       = cf.dt_vector(dt_str_p, calendar=calendar)  # t-coordinate
    #
    # cidp = np.array([p[-1] for p in tracks])                        # cyclone track id
    #
    # # remove first 5 transitional years
    # mask5years = (tp < date_start)
    # yp   = yp[  ~mask5years]
    # xp   = xp[  ~mask5years]
    # tp   = tp[  ~mask5years]
    # cidp = cidp[~mask5years]
    #
    # # temporal range
    # date_ini = tp[ 0]
    # date_end = tp[-1]
    #
    # # convert units of x-y coordinates (from km to m)
    # yp *= 1000
    # xp *= 1000

    # number of processed years
    nyrs = int(date_end.strftime('%Y')) - int(date_ini.strftime('%Y')) + 1

    # calculate frequency of cyclonic activity
    # ________________________________________

    # logging message
    indent = log_message(f'[{simid}] calculating cyclonic ocurrence')

    # initial computational time
    tref_ini = dt.datetime.now()

    # create iterator of lat-lon indexes (only works because len(lon) > len(lat))
    args = (ij for ij in grid_index_iterator(x, y))

    # create frequency container
    count = np.zeros(len(x) * len(y))

    # create thread pool
    with mp.Pool(processes=4) as pool:

        # compute processing
        results = pool.imap(count_tracks, args)

        # fill container
        for k, count_ij in enumerate(results):

            # update values inside container
            count[k] = count_ij

    # reshape container
    count.shape = (len(x), len(y))

    # create xarray for cyclone count
    count = xr.DataArray(data=count / nyrs,
                        name='count',
                        coords={'x': x, 'y': y},
                        dims=['x', 'y'],
                        attrs={'long_name': 'Cyclonic birth frequency',
                                'units'    : 'per year'})

    # add border dates inside attributes
    count.attrs['date_start'] = date_ini.strftime('%Y-%m-%d %H:%M')
    count.attrs['date_end'  ] = date_end.strftime('%Y-%m-%d %H:%M')

    # # save results
    count.to_netcdf(f'{dirdata}/cyclonic_ini_ocurrences_{simid.lower()}.nc')

    # final computational time
    tref_end = dt.datetime.now()

    # computational timedelta
    tref_delta = (tref_end - tref_ini) / dt.timedelta(minutes=1)

    # logging message (regarding computational time)
    print(f'{indent}computational time: {tref_delta:.2f}')

# logging message
indent = log_message(f'[{simid}] creating storm track birth density map')

# update output directory and create if it doesn't exists
os.makedirs(f'{dirimg}') if not os.path.isdir(f'{dirimg}') else None

# # check if pis zoom is needed
# if 'pis' in pis_sa:
#
#     # open patagonian ice sheet spatial mask
#     pismask = xr.open_dataset(f'{dirdata}/pis_mask.nc')['ice']
#
#     # fix longitude to be cyclic
#     lon = pismask['lon'].data
#     lon[-1] = 360
#     pismask['lon'] = pismask['lon'].where(False, lon)
#
#     # convert coordinate system (from lat-lon to x-y)
#     pismask = convert_to_lambert(pismask, method='nearest')
#
#     # convert coordinates units (from km to m) (only 1D)
#     pismask['x'] = pismask['x'].where(False, pismask['x'] * 1000)
#     pismask['y'] = pismask['y'].where(False, pismask['y'] * 1000)
#
#     pismask['X'] = pismask['X'].where(False, pismask['X'] * 1000)
#     pismask['Y'] = pismask['Y'].where(False, pismask['Y'] * 1000)
#
#     # extract x-y coordinates (1D)
#     x = pismask['x'].data
#     y = pismask['y'].data
#
#     # # expand pismask
#     # pismask_expanded = expand_pismask(pismask, D=840000)

# load base simulation
count = xr.open_dataset(f'{dirdata}/cyclonic_ini_ocurrences_{simid}.nc')['count']
base  = xr.open_dataset(f'{dirdata}/cyclonic_ini_ocurrences_lgm_100.nc')['count']

# output filename template
output = f'piss_{anom}map{pis_sa}_ini_freq_{simid.lower()}{nocoast}.png'

# smooth fields
count = smooth_array(count, r)
base  = smooth_array(base , r)

# calculate anomaly
if anom:

    count = count.where(False, count - base)
    count.attrs['long_name'] += ' anomaly' if anom else ''

# clean "base" variable
del base

date_ini = cf.dt(count.attrs['date_start'], calendar=calendar)
date_end = cf.dt(count.attrs['date_end']  , calendar=calendar)

# text box parameters
bbox = {'boxstyle' : 'round',
        'facecolor': 'wheat',
        'edgecolor': 'black',
        'alpha'    : 0.5}

# create temporal information box
datestr  = f"Years : {date_ini.strftime('%Y')} - {date_end.strftime('%Y')}"
datestr += f"\nSeason: year-round"

# figure parameters
if anom:

    vmin = -3.5
    vmax =  13.5

    start = 0
    stop  = 1
    midp  = 1 - (vmax / (vmax + abs(vmin)))

    cmap = cm.RdBu_r
    cmap = shiftedColorMap(cmap, start=start, stop=stop, midpoint=midp)

    extend = 'both'

    # contour levels to draw
    clev = np.array([-5, 5])

    # contourf levels to color
    levels = np.arange(vmin, vmax, 1)

else:

    vmin = 0
    # vmax = 60 if (simid != 'ra') else 200
    vmax = 60

    start = 0
    stop  = 1
    midp  = 0.5

    # cmap = cm.hot_r
    # cmap = shiftedColorMap(cmap, start=start, stop=stop, midpoint=midp)
    cmap = cmaps.WhiteBlueGreenYellowRed

    extend = 'max'

    # contour levels to draw
    clev   = np.array([10])

    # contourf levels to color
    levels = np.arange(vmin, vmax+1, 4)

# axis parameters to make it circular
if 'pis' in pis_sa:

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

    colorbar_kwargs = {# 'ticks' : levels[::3],
                       # 'shrink': 0.75,
                       'orientation': 'horizontal',
                       'pad'        : -1,
                       'aspect'     : 30,
                       'shrink'     : 0.3}

else:

    theta  = np.linspace(0, 2*np.pi, 100)
    center = [0.5, 0.5]
    radius = 0.5
    verts  = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)

    datestr_pos = [0.78, 0.0]

    colorbar_kwargs = {# 'ticks' : levels[::3],
                       # 'shrink': 0.75,
                       'orientation': 'vertical',
                       'pad': 0.08}

if ('sa' in pis_sa):

    datestr_pos = [0.78, -0.1]

# scale factor for pis grid markers
pis_sa_ms  = 1     if pis_sa else 0.5
pis_sa_res = '50m' if pis_sa else '110m'

# define crop command
crop_cmd = f'{homedir}/.conda/envs/conda/bin/convert -crop 1250x900+120+0'

# create figure
fig, ax = plt.subplots(1, 1, figsize=size, subplot_kw={'projection': proj})

count_plot = partial(count.plot.contourf)

# plot field
count_plot(ax=ax,
           x='x',
           transform=transxy,
           # vmin=vmin,
           # vmax=vmax,
           levels=levels,
           extend=extend,
           add_colorbar=True,
           cmap=cmap,
           cbar_kwargs=colorbar_kwargs)

# show important contour levels
cont = count.plot.contour(ax=ax,
                          x='x',
                          transform=transxy,
                          levels=clev,
                          colors='black',
                          add_colorbar=False)

ax.clabel(cont,
          inline=True,
          fontsize='x-small',
          inline_spacing=1) if (not pis_sa) else None

# add coastlines
ax.coastlines(color='black',
              linewidth=0.5,
              resolution=pis_sa_res) if (not nocoast) else None

# add temporal info. box
ax.text(*datestr_pos, datestr, bbox=bbox, size=7, family='monospace',
        transform=ax.transAxes)

# add and adjust gridlines
gridlines = ax.gridlines(linewidth=1,
                         color='grey',
                         alpha=0.1,
                         ls='--',
                         draw_labels=False,
                         x_inline =False,
                         y_inline =False)

# gridlines.xlocator = mticker.MultipleLocator(30)

gridlines.xlabels_top    = False
gridlines.xlabels_bottom = False
gridlines.ylabels_left   = True
gridlines.ylabels_right  = False

# set labels
title  = f'{simid.upper()}'
title += ' (anomaly)' if anom else ''
ax.set_title(title)
ax.set_xlabel('')
ax.set_ylabel('')

if (not pis_sa) or ('pis' in pis_sa):

    # make axis circular
    ax.set_boundary(circle, transform=ax.transAxes)

    # set coordinate limits
    ax.set_extent([0, 360, -90, -30], crs=trans)

else:

    # set coordinate limits [W, E, S, N]
    ax.set_extent([275, 310, -60, -30], crs=trans)

# logging message
print(f'{indent}{output}')

# save / show plot
fig.tight_layout() if (not 'sa' in pis_sa) else None
fig.savefig(f"{dirimg}/{output}", bbox_inches='tight')
plt.close()

# crop temporal pdf to remove unnecessary headers
if ('pis' in pis_sa):

    os.system(f'{crop_cmd} {dirimg}/{output} {dirimg}/{output} > /dev/null')


# final logging message
print('\n:: uwu\n\n')


























