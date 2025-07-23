import xarray as xr
import numpy as np
from numpy import empty
import pftnames

''' -------------------------------------------------------------------------------------------------
    Python function to convert PFT formatted CLM output to gridded output using area maps for CFT. 
    Adapted from:https://github.com/NCAR/ctsm_python_gallery/blob/master/notebooks/PFT-Gridding.ipynb
    ------------------------------------------------------------------------------------------------- '''
def reshape_1D_to_2Dgrid(varname, data_in, mask =None):
    ixy      = data_in.pfts1d_ixy
    jxy      = data_in.pfts1d_jxy
    landfrac = data_in.landfrac
    lat      = data_in.lat
    lon      = data_in.lon
    time     = data_in.time
    vegtype  = data_in.pfts1d_itype_veg
    pfts     = np.array(pftnames.pftname)

    ntim  = len(time)
    nlat  = len(lat)
    nlon  = len(lon)
    npfts = len(pfts)

    var = data_in[varname]

    # Create empty array
    gridded = empty([ntim, npfts, nlat, nlon])

    # Fill using PFT, jxy (lat), ixy (lon)
    gridded[:, vegtype.values.astype(int),
    jxy.values.astype(int) - 1,
    ixy.values.astype(int) - 1] = var.values

    # Create DataArray with coordinates and dims
    grid_dims = xr.DataArray(
        gridded,
        dims=("time", "pft", "lat", "lon"),
        coords=dict(time=time, pft=pfts, lat=lat.values, lon=lon.values),
        name=varname
    )

    # Apply land fraction mask (assumes landfrac has shape (lat, lon))
    grid_dims = grid_dims.where(landfrac == True)

    grid_dims = grid_dims.isel(pft=slice(15, 79))

    if mask is not None:
        mask = mask.assign_coords({
        'lat': grid_dims.lat,
        'lon': grid_dims.lon,
        'time': grid_dims.time,
        'pft': grid_dims.pft
            })
        grid_dims        = grid_dims.where(mask)

    return grid_dims