import xarray as xr
import numpy as np

def calc_climatology(ds, ref_years=(1981, 2010)):
    '''
    Compute the climatological mean (TXnorm) of daily maximum temperatures using 
    a 15-day rolling window centered on each calendar day.

    Parameters:
    tx (xarray.DataArray): Daily maximum temperature with dimensions ('time', 'lat', 'lon').
    ref_years (tuple): Reference period (start_year, end_year) for climatology.

    Returns:
    xarray.DataArray: TXnorm with a climatological mean for each calendar day.
    '''
    # Apply a 5-day rolling mean centered on each day
    ds_rollmean = ds.rolling(time=15, center=True, min_periods=1).construct("window")
    # Select reference period
    ds_ref = ds_rollmean.sel(time=slice(f"{ref_years[0]}-01-01", f"{ref_years[1]}-12-31"))
    # Convert time to day-of-year (DOY)
    doy    = ds_ref.time.dt.dayofyear
    # Compute mean for each day of the year across all reference years
    climatology = ds_ref.groupby(doy).mean(dim=("window", "time"))  

    # Ensure only 365 days remain
    climatology = climatology.sel(dayofyear=climatology["dayofyear"] != 60)
    return climatology

def HWmask_climanomaly(ds, climatology,Textr=5):
    doy    = ds.time.dt.dayofyear
    anomaly_extr = ds.groupby(doy) - (climatology + Textr)
    #extreme years
    mask_extremes  = np.where((anomaly_extr > 0) & (ds>= 28),1, np.nan)
    return mask_extremes

def onset_filter(array, axis):
    """
    Returns locations when the values of array go from nan to valid for at least 3 steps

    This should be used with a centred, 5 element window like
        x.rolling(time=5, center=True, min_periods=1).reduce(rising_filter)

    Note that applying this to a Dask array will load the entire input array
    source: https://gist.github.com/ScottWales/dd9358bea2547c99e46b197bc9f53d21
    """
    # Make sure there are enough points
    assert(array.shape[axis] == 5)
    # Make sure we're working on the last axis
    assert(axis == array.ndim-1 or axis == -1)

    left = array[..., 1]
    right = array[..., 2:].sum(axis=axis)

    return np.logical_and(np.isnan(left), np.isfinite(right))

def offset_filter(array, axis):
    """
    Returns locations when the values of array go from valid to nan for at least 3 steps

    This should be used with a centred, 5 element window like
        x.rolling(time=5, center=True, min_periods=1).reduce(rising_filter)

    Note that applying this to a Dask array will load the entire input array
    """
    # Make sure there are enough points
    assert(array.shape[axis] == 5)
    # Make sure we're working on the last axis
    assert(axis == array.ndim-1 or axis == -1)

    left = array[..., :3].sum(axis=axis)  # Sum the first 3 elements (should be valid)
    right = array[..., 3]  # Fourth element

    return np.logical_and(np.isfinite(left), np.isnan(right))

def HW_mask(temperature, HW_definition='anom'):
    # --------- 1. define heat stress using temperature dataset and HS definition
    if HW_definition == 'anom':
        # define heatstress
        clim     = calc_climatology(temperature)
        # Drop the 29th of February (DOY = 60)
        #clim_365 = clim.sel(dayofyear=clim.dayofyear[clim.dayofyear != 60])

        MASK_HW  = HWmask_climanomaly(temperature, clim)
    else:
        print('heatwave definition unknown')

    MASK_HW      = xr.DataArray(MASK_HW, dims=["time", "lat", "lon"], coords={"time": temperature['time'], "lat": temperature['lat'], "lon": temperature['lon']})



    # --------- 2. calculate onset, offset and duration
    hwstart      = MASK_HW.rolling(time=5, center=True, min_periods=1).reduce(onset_filter, 0)
    hwend        = MASK_HW.rolling(time=5, center=True, min_periods=1).reduce(offset_filter, 0)



    # ---------  3. generate HW mask dataset
    #             Create a mask where there is at least one '1' over time
    valid_mask = (MASK_HW == 1).any(dim="time")
    # Get the (lat, lon) indices where valid_mask is True
    lat_idx, lon_idx = np.where(valid_mask)
    # Convert indices to list of tuples
    valid_locations = list(zip(lat_idx, lon_idx))
    len(valid_locations)

    # define empty np.array
    times      = temperature['time'].values
    lats       = temperature['lat'].values
    lons       = temperature['lon'].values

    dims       = (len(times),len(lats), len(lons))
    HW_lengths = np.full(dims, np.nan)

    #                calculate duration of heatwaves for each cell
    for i in valid_locations:
        #print(i)
        arr_onset   = hwstart[:,i[0],i[1]]
        arr_offset  = hwend[:,i[0],i[1]]
        
        onset_drop  = arr_onset.where(arr_onset ==1, drop=True)
        offset_drop = arr_offset.where(arr_offset ==1, drop=True)
        assert offset_drop.shape[0] == onset_drop.shape[0]
    
        # get duration of heatwave in days
        HW_durations    = ((offset_drop['time'].values -onset_drop['time'].values).astype("timedelta64[D]").astype(int)+1)
        #regain timestamps
        ds_HW_durations = xr.DataArray(HW_durations, coords={"time": onset_drop["time"]}, dims="time")
        # Reindex `test` to match HW_lengths' time dimension, filling gaps with NaN
        ds_HW_aligned   = ds_HW_durations.reindex(time=arr_onset["time"], fill_value=np.nan)
        #give it index arr_onset where MASK_HW_
        HW_lengths[:,i[0],i[1]] = ds_HW_aligned
    
    #                     output data
    ds_HW = xr.Dataset(
        {
            "HW_onset": (["time", "lat", "lon"], hwstart.values),
            "HW_offset": (["time", "lat", "lon"], hwend.values),
            "HW_lengths": (["time", "lat", "lon"], HW_lengths),
        },
        coords={"time": times, "lat": lats, "lon": lons},
    )

    
    return ds_HW
