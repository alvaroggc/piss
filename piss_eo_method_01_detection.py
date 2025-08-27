'''

  Software for the tracking of storms and high-pressure systems

'''

#
# Load required modules
#

import numpy as np
from datetime import date
from netCDF4 import Dataset
import xarray as xr

from matplotlib import pyplot as plt

import storm_functions as storm


def load_simulation_variable(dirin, simid, key='PSL', cyclic=False):
    '''
    info:
    parameters:
    returns:
    '''

    # check files to load
    if (key in ['PSL']):

        # list of netcdf files to load
        fin = sorted(glob(f'{dirin}/data/piss/yData/h1/{simid}/{simid}.*.nc'))

    elif (key in ['PHIS']):

        # list of netcdf files to load
        fin = sorted(glob(f'{dirin}/data/piss/phis/{simid}.*.nc'))

    # logging message
    indent = log_message(f'loading {key}')
    for f in fin: print(f"{indent}{f.split('/')[-1]}")

    # load all files inside dataset container
    da = xr.open_mfdataset(fin,
                           compat='override',
                           coords='minimal')[key]

    # round spatial coordinates
    da['lat'] = da['lat'].astype(np.float32)
    da['lon'] = da['lon'].astype(np.float32)

    # make longitude cyclic
    if cyclic:

        # clone first value to last position
        dalast        = da.sel({'lon': 0}).copy()
        dalast['lon'] = 360
        da            = xr.concat((da, dalast), 'lon')

    # output
    return da



#
# Load in slp data and lat/lon coordinates
#

homedir = os.path.expanduser('~')               # home directory
dirsim  = f'/mnt/cirrus/results/friquelme'      # cesm simulations
dirout  = f'{homedir}/projects/piss/data'       # data output

simid = 'lgm_100'

# load simulation datasets
slp = load_simulation_variable(dirsim, simid, 'PSL', cyclic=True)

# separate coordinates
lat = slp['lat'].data
lon = slp['lon'].data

input('stop')

# Storm Detection
#

# Initialisation

lon_storms_a = []
lat_storms_a = []
amp_storms_a = []
lon_storms_c = []
lat_storms_c = []
amp_storms_c = []

# Loop over time

T = slp.shape[0]

for tt in range(T):
    #
    print tt, T
    #
    # Detect lon and lat coordinates of storms
    #
    lon_storms, lat_storms, amp = storm.detect_storms(slp[tt,:,:], lon, lat, res=2, Npix_min=9, cyc='anticyclonic', globe=True)
    lon_storms_a.append(lon_storms)
    lat_storms_a.append(lat_storms)
    amp_storms_a.append(amp)
    #
    lon_storms, lat_storms, amp = storm.detect_storms(slp[tt,:,:], lon, lat, res=2, Npix_min=9, cyc='cyclonic', globe=True)
    lon_storms_c.append(lon_storms)
    lat_storms_c.append(lat_storms)
    amp_storms_c.append(amp)
    #
    # Save as we go
    #
    if (np.mod(tt, 100) == 0) + (tt == T-1):
        print 'Save data...'
    #
    # Combine storm information from all days into a list, and save
    #
        storms = storm.storms_list(lon_storms_a, lat_storms_a, amp_storms_a, lon_storms_c, lat_storms_c, amp_storms_c)
        np.savez('storm_det_slp', storms=storms, year=year, month=month, day=day, hour=hour)

