#!/home/cr2/agomez/.conda/envs/conda/bin/python
# -*- coding: utf-8 -*-
#
# ~/projects/piss/src/piss_map_01_slp_minimums.py
#
# info  : plot maps with slp minimums locations
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
dirimg  = f'{homedir}/projects/piss/img2'       # output for figures

# output file with cyclones information
fin = f'cyclones_{simid}_v3_0001_0001.pkl'

# indexer to choose southern hemisphere (from -30° to the south)
shidx = {'lat': slice(-90, -30)}

# parameters needed in method
topo_max = 1000     # slp minimum points over this altitude are eliminated [m]
radius   = 1000     # gradient search radius [km]
grad_min = 10       # minimum slp gradient required for cyclone [hPa]

# figure parameters
size = (7.5, 6)


###################
##### RUNNING #####
###################


# open cyclones file
with open(f'{dirdata}/{fin}', 'rb') as f:

    # read file
    cyclones = pickle.load(f)

# temporal range
date_ini = [*cyclones.keys()][ 0]
date_end = [*cyclones.keys()][-1]

# load simulation datasets
slp = load_simulation(dirsim, simid, ['PSL'])['PSL']

# extract temporal range (only for slp)
slp = slp.sel({'time': slice(date_ini, date_end)})

# adjust slp units
slp = slp.where(False, slp / 100)
slp.attrs['units'] = 'hPa'

# output filename template
output_template = f'piss_map_slpmin_v3_{simid.lower()}_*.png'

# remove previous images
for f in glob(f'{dirimg}/{output_template}'): os.remove(f)

# levels of slp (for maps)
levels = np.arange(970, 1050+1, 5)

# logging message
indent = log_message('making slpmin maps')

# plot tracks over map
for t in slp['time'][:181]:

    # date identifiers
    datestr = t.dt.strftime('%Y-%m-%d').item()
    dateid  = t.dt.strftime( '%Y%m%d' ).item()

    # output filename image
    output = f"{output_template.replace('*', dateid)}"

    # logging message
    print(f'{indent}{output}')

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

    # plot minimum points
    for p in cyclones[datestr]:

        # transform to lat-lon coordinates
        plat, plon = convert_point_to_latlon(p)

        # draw point in map
        ax.plot(plon, plat, lw=1, transform=trans,
                marker='o', mfc='yellow', mec='black', zorder=4)#, ms=5)

        # write code of slp minimum
        ax.text(plon, plat, p[-1], size=12, transform=trans, zorder=5, color='gold')

        # # extract border points of search radius
        # latb = [pb[0] for pb in p[5]]
        # lonb = [pb[1] for pb in p[5]]
        #
        # # draw border points of search radius
        # ax.plot(lonb, latb, lw=0, transform=trans,
        #         marker='.', ms=1, mec='snow', mfc='snow')

    # add coastlines
    ax.coastlines(color='black', linewidth=0.5)

    # add and adjust gridlines
    gridlines = ax.gridlines(linewidth=1,
                                color='grey',
                                alpha=0.25,
                                ls='--')

    gridlines.top_labels    = True
    gridlines.bottom_labels = True
    gridlines.left_labels   = True
    gridlines.right_labels  = True

    # set labels
    ax.set_title(f'Southern Hemishphere, {datestr}')
    ax.set_xlabel('')
    ax.set_ylabel('')

    # remove box
    # ax.axis('off')

    # set coordinate limits
    ax.set_extent([0, 360, -90, -9], crs=trans)

    # save / show plot
    fig.savefig(f"{dirimg}/{output}", bbox_inches='tight')

    # close figure
    plt.close()

