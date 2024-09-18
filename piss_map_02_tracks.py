#!/home/cr2/agomez/.conda/envs/conda/bin/python
# -*- coding: utf-8 -*-
#
# ~/projects/piss/src/piss_map_02_tracks.py
#
# info  : plot cyclone tracks
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
import matplotlib.path as mpath
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
fin = f'tracks_{simid}_v3_0001_0005.pkl'

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
    tracks = pickle.load(f)

# list of tracks IDs
tids = [*tracks.keys()]

# load simulation datasets
slp = load_simulation(dirsim, simid, ['PSL'], cyclic=True)['PSL']

# adjust slp units
slp = slp.where(False, slp / 100)
slp.attrs['units'] = 'hPa'

# output filename template
output_template = f'tracks/piss_map_tracks_v3_{simid.lower()}_*.png'

# remove previous images
for f in glob(f'{dirimg}/{output_template}'): os.remove(f)

# levels of slp (for maps)
levels = np.arange(970, 1050+1, 5)

# axis parameters to make it circular
theta  = np.linspace(0, 2*np.pi, 100)
center = [0.5, 0.5]
radius = 0.5
verts  = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)

# logging message
indent = log_message('making cyclone tracks')

# plot tracks over map
for tid in tids[:10]:

    # output filename image
    output = f"{output_template.replace('*', tid)}"

    # logging message
    print(f"{indent}{output.split('/')[-1]}")

    # temporal range
    date_ini = tracks[tid][ 0][-2]
    date_end = tracks[tid][-1][-2]

    # time indexer
    tidx = {'time': slice(date_ini, date_end)}

    # calculate slp mean through tracks
    slp_track = slp.sel(tidx).mean(dim='time', keep_attrs=True)

    # create figure
    fig, ax = plt.subplots(1, 1, figsize=size,
                           subplot_kw={'projection': proj})

    # plot field
    slp_track.plot.contourf(ax=ax,
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
    for p in tracks[tid]:

        # separate contour coordinates
        xc, yc = p[4][1:]

        # convert units
        xc = xc * 1000
        yc = yc * 1000

        # transform to lat-lon coordinates
        plat, plon = convert_point_to_latlon(p)

        # draw point in map
        ax.plot(plon, plat, lw=1, transform=trans,
                marker='o', mfc='yellow', mec='black', zorder=4)#, ms=5)

        # write code of slp minimum
        ax.text(plon, plat, p[-1], size=12,
                transform=trans, zorder=5, color='gold')

        # # draw contour
        # ax.plot(xc, yc, lw=1, color='black', transform=transxy)

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
    ax.set_title(f'Southern Hemishphere, track {tid}')
    ax.set_xlabel('')
    ax.set_ylabel('')

    # remove box
    # ax.axis('off')

    # make axis circular
    ax.set_boundary(circle, transform=ax.transAxes)

    # set coordinate limits
    ax.set_extent([0, 360, -90, -30], crs=trans)

    # save / show plot
    fig.savefig(f"{dirimg}/{output}", bbox_inches='tight')

    # close figure
    plt.close()

