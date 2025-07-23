import xarray as xr
import numpy as np

''' functions used to define heatwaves.
    -------------------- CLIMATOLOGY -----------------------------------------------------------------------------
    calc_clim           : calculates climatology of reference period
    calc_clim_sd        : same as calc clim but including standard deviation
    -------------------- HEATWAVE DEFINITIONS --------------------------------------------------------------------
    HWmask_clim_ABSanom : mask based on absolute threshold exceeding the climatology (kelvin or celsius)
    HWmask_clim_SDanom  : mask based on exceeding the standard deviation of reference climatology (kelvin or celcius)
    -------------------- HEATWAVE OUTPUT ------------------------------------------------------------------------
    HW_mask             : generates output dataset calculating the occurence and duration of these HW definitions 
    '''

def calc_clim(ds, window, ref_years=(1961, 1990)):
    '''
    Compute the climatological mean of temperature dataset (TXnorm) using
    a x-day rolling window centered on each calendar day.

    Parameters:
    tx (xarray.DataArray): Temperature array with dimensions ('time', 'lat', 'lon').
    ref_years (tuple)    : Reference period (start_year, end_year) for climatology.
    window               : number of days to include around DOY for smoothing

    Returns:
    xarray.DataArray: TXnorm with a climatological mean for each calendar day.
    '''
    # Apply a 15-day rolling mean centered on each day
    ds_rollmean = ds.rolling(time=window, center=True,
                             min_periods=1).construct("window")
    # Select reference period
    ds_ref = ds_rollmean.sel(time=slice(f"{ref_years[0]}-01-01",
                                        f"{ref_years[1]}-12-31"))

    # Convert time to day-of-year (DOY)
    doy = ds_ref.time.dt.dayofyear

    # Compute mean for each day of the year across all reference years
    climatology = ds_ref.groupby(doy).mean(dim=("window", "time"))

    # Ensure only 365 days remain
    if doy.max() == 366:
        climatology = climatology.sel(dayofyear=climatology["dayofyear"] != 60)

    return climatology


def calc_clim_sd(ds, window, ref_years=(1961, 1990)):
    """
    Compute climatological mean and std of temperature with a x-day rolling window,
    grouped by day of year (DOY).

    Parameters:
    ds (xarray.DataArray): Temperature data with 'time' dimension.
    ref_years (tuple)    : Reference period (start_year, end_year).
    window               : number of days to include around DOY for smoothing

    Returns:
    (clim_mean, clim_std): Two DataArrays of shape (dayofyear, lat, lon).
    """
    ds_ref = ds.sel(time=slice(f"{ref_years[0]}-01-01",
                               f"{ref_years[1]}-12-31"))

    # Apply 15-day rolling and construct window
    ds_roll = ds_ref.rolling(time=window, center=True,
                             min_periods=1).construct("window")

    # Assign DOY for grouping
    doy = ds_roll.time.dt.dayofyear
    ds_roll.coords["doy"] = ("time", doy.data)

    # Compute mean and std across time and window
    clim_mean = ds_roll.groupby("doy").mean(dim=("time", "window"))
    clim_std = ds_roll.groupby("doy").std(dim=("time", "window"))

    # Drop leap day (Feb 29)
    if doy.max() == 366:
        clim_mean = clim_mean.sel(doy=clim_mean["doy"] != 60)
        clim_std = clim_std.sel(doy=clim_std["doy"] != 60)

    return clim_mean, clim_std


def HWmask_clim_ABSanom(ds, climatology, Tmin=28, Textr=5):
    doy = ds.time.dt.dayofyear

    anomaly_extr = ds.groupby(doy) - (climatology + Textr)

    # extreme years
    mask_extremes = np.where((anomaly_extr > 0) & (ds >= Tmin), 1, np.nan)

    return mask_extremes


def HWmask_clim_SDanom(ds, clim_mean, clim_std, Tmin=27, SD_mf=1.5):
    """
    Detect heatwave extremes when temperature exceeds climatology + 2*std and a Tmin threshold.

    Parameters:
    ds (xarray.DataArray): Daily temperature dataset.
    clim_mean (xarray.DataArray): Climatological mean (dayofyear, lat, lon).
    clim_std  (xarray.DataArray): Climatological std (dayofyear, lat, lon).
    Tmin (float): Minimum temperature threshold for defining a heatwave.
    SD (float)  : Multiplication factor for standard deviation to define anomaly upper tail
                  Default is 1.5 (~93 percentile), consider: 1.75 (96th), 2.0 (97.7th), 3.0 (99.7th)

    Returns:
    xarray.DataArray: Binary mask (1 = extreme, nan = non-extreme).
    """

    doy              = ds.time.dt.dayofyear
    ds.coords["doy"] = doy

    threshold        = clim_mean + SD_mf * clim_std
    # Align threshold to each day by indexing using doy
    threshold_for_day = threshold.sel(doy=ds.doy)

    # Apply extreme condition
    mask_extremes = xr.where((ds > threshold_for_day) & (ds >= Tmin), 1, np.nan)

    return mask_extremes


def HW_mask(temperature, clim, SD, factor, tmin=28,HW_definition='SD_anom'):
    """
    Uses temperature to define the timing and duration of heatwaves, based on the heatwave definition.
    parameters:
    - temperature : xarray dataset (noleap in cftime) on which the heatwaves must be computed
    - HW_definition: current options are: abs_anom (based on a fixed temperature threshold above the climatology) and
                    SD_anom (based on standard deviation above climatological mean)
    - clim        : climatology (doy,lat,lon) based on previously indicated temperature (same variable as temperature)
    - SD          : standard deviation of clim (doy,lat,lon)
    - tmin        : minimum threshold that must be exceeded for anomaly temperature to be considered as heat stress
    - factor      : either absolute temperature to be exceeded (abs_anom) or multiplication factor of SD (SD_anom)

    Returns a dataset with 3 variables:
    - Heatwave onsets
    - Heatwave offset
    - Heatwave lengths (indicated on the day of HW onset)
    """

    # --------- 1. Create mask in which heatwave definition when true is set to 1
    if HW_definition == 'abs_anom':
        # calculate climatology
        MASK_HW      = HWmask_clim_ABSanom(temperature, clim, Tmin=tmin, Textr=factor)
    elif HW_definition =='SD_anom':
        MASK_HW        = HWmask_clim_SDanom(temperature,clim, SD, Tmin=tmin, SD_mf=factor)
    else:
        print('heatwave definition unknown')
    MASK_HW = xr.DataArray(MASK_HW, dims=["time", "lat", "lon"],
                           coords={"time": temperature['time'], "lat": temperature['lat'], "lon": temperature['lon']})

    #             close redundant dataset to free memory
    temperature.close()

    # --------- 2. calculate onset, offset and duration heatwaves
    hwstart = MASK_HW.rolling(time=5, center=True, min_periods=1).reduce(onset_filter, 0)
    hwend = MASK_HW.rolling(time=5, center=True, min_periods=1).reduce(offset_filter, 0)

    #             Create a mask where there is at least one '1' over time
    valid_mask = (MASK_HW == 1).any(dim="time")
    lat_idx, lon_idx = np.where(valid_mask)
    valid_locations = list(zip(lat_idx, lon_idx))

    #             define empty np.array
    times = MASK_HW['time'].values
    lats = MASK_HW['lat'].values
    lons = MASK_HW['lon'].values

    dims = (len(times), len(lats), len(lons))
    HW_lengths = np.full(dims, np.nan)

    #             close redundant dataset to free memory
    MASK_HW.close()

    #             calculate duration of heatwaves for each cell
    for i in valid_locations:
        # define onset & offsets heatwaves
        arr_onset  = hwstart[:, i[0], i[1]]
        arr_offset = hwend[:, i[0], i[1]]

        onset_drop  = arr_onset.where(arr_onset == 1, drop=True)
        offset_drop = arr_offset.where(arr_offset == 1, drop=True)
        assert offset_drop.shape[0] == onset_drop.shape[0]

        # get duration of heatwave in days
        HW_durations     = ((offset_drop['time'].values - onset_drop['time'].values).
                            astype("timedelta64[D]").astype(int) + 1)
        # regain timestamps
        ds_HW_durations  = xr.DataArray(HW_durations,
                                       coords={"time": onset_drop["time"]},
                                       dims="time")

        # Reindex `test` to match HW_lengths' time dimension, filling gaps with NaN
        ds_HW_aligned    = ds_HW_durations.reindex(time=arr_onset["time"],
                                                fill_value=np.nan)

        # give it index arr_onset where MASK_HW
        HW_lengths[:, i[0], i[1]] = ds_HW_aligned

    # ---------  3. generate HW mask dataset
    ds_HW = xr.Dataset(
        {
            "HW_onset": (["time", "lat", "lon"], hwstart.values),
            "HW_offset": (["time", "lat", "lon"], hwend.values),
            "HW_lengths": (["time", "lat", "lon"], HW_lengths),
        },
        coords={"time": times, "lat": lats, "lon": lons},
    )

    return ds_HW

def onset_filter(array, axis):
    """
    Returns locations when the values of array go from nan to valid for at least 3 steps

    This should be used with a centred, 5 element window like
        x.rolling(time=5, center=True, min_periods=1).reduce(rising_filter)

    Note that applying this to a Dask array will load the entire input array

    source: https://gist.github.com/ScottWales/dd9358bea2547c99e46b197bc9f53d21
    """
    # Make sure there are enough points
    assert (array.shape[axis] == 5)

    # Make sure we're working on the last axis
    assert (axis == array.ndim - 1 or axis == -1)

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
    assert (array.shape[axis] == 5)

    # Make sure we're working on the last axis
    assert (axis == array.ndim - 1 or axis == -1)

    left = array[..., :3].sum(axis=axis)  # Sum the first 3 elements
    right = array[..., 3]  # Fourth element

    return np.logical_and(np.isfinite(left), np.isnan(right))
