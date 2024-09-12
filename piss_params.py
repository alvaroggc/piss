#!/home/cr2/agomez/.conda/envs/conda/bin/python
# -*- coding: utf-8 -*-
#
# ~/cesm/lib/thesis_parameters
#
# info  : common parameters used for processing CESM simulations.
# author: @alvaroggc


# 3rd party packages
import cf
import pyproj
import numpy as np
import cartopy.crs as ccrs



#############################
##### GLOBAL PARAMETERS #####
#############################


# projection definitions (laea = lambert azimuthal equal-area projection)
latlon_crs = 'epsg:4326'
laea_crs   = '+proj=laea +lon_0=-73.53 +lat_0=-90 +ellps=WGS84 +x_0=0 +y_0=0 +units=km'

# map definition
trans = ccrs.PlateCarree()
proj  = ccrs.NearsidePerspective(central_longitude=(360-73.53),
                                 central_latitude=-90)

# datetime normalization parameters
ref_datetime = cf.dt(2000, 1, 1, calendar='gregorian')
norm         = np.timedelta64(1, 'D')




