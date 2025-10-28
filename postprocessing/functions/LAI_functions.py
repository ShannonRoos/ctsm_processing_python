import xarray as xr
import numpy as np


def calc_LAI_mean(ds_cgls):
    '''
    Compute the LAI mean of 10-daily LAI using
    a 15-day rolling window centered on each calendar day.

    Parameters:
    ds_cgls (xarray.DataArray): x-daily LAI ('time', 'lat', 'lon').

    Returns:
    xarray.DataArray: smoothed mean LAI representing climatological mean for each calendar day.
    '''
    #go from dekads to daily data
    ds_daily = ds.resample(time="1D").bfill()
    # Apply a 15-day rolling mean centered on each day
    ds_rollmean = ds_daily.rolling(time=15, center=True, min_periods=1).construct("window")
    # Convert time to day-of-year (DOY)
    doy    = ds_rollmean.time.dt.dayofyear
    # Compute mean for each day of the year across all reference years
    mean_LAI = ds_rollmean.groupby(doy).mean(dim=("window", "time"))
    return  mean_LAI

def slope_filter(arr,days=3):
    # Assume `slope` is your dataset with NaNs for masked-out values
    slope_binary = ~np.isnan(arr)  # Convert to binary (1 for valid, 0 for NaN)

    # Label consecutive groups of valid values
    labeled_array, num_features = label(slope_binary)

    # Count occurrences of each label
    label_counts = np.bincount(labeled_array.ravel())

    # Create a mask keeping only groups with 3 or more consecutive values
    valid_labels = np.where(label_counts >= days)[0]
    valid_mask = np.isin(labeled_array, valid_labels)

    # Apply the mask to keep only valid slope values
    slope_filtered = arr.where(valid_mask)
    return slope_filtered


def calc_lai_slopes(LAI, tdim='time'):
    LAI_diff    = LAI.diff(dim=tdim)
    # make sure slope values are on the starting date and not the ending date
    difftime    = LAI[tdim][:-1].values
    diff_tshift = LAI_diff.assign_coords(time=(tdim, difftime))
    lai_dek     = LAI[:-1, :,:]
    lai_dek_ann = lai_dek.groupby(lai_dek.time.dt.year)
    annual_max  = lai_dek_ann.max().mean()
    # mask out non-sensescence
    mask = np.where(((diff_tshift < 0 ) & (lai_dek> 0.1*annual_max) ), 1,0)
    lai_filtered = lai_dek * mask
    slope = lai_filtered.where(lai_filtered > 0)
    slope_masked = slope_filter(slope,8)
    return slope_masked

def count_valid_sequences(arr):
    """Returns the count of consecutive non-NaN sequences with at least 3 values."""
    isnan = np.isnan(arr)
    seq_start = ~isnan & np.roll(isnan, 1)
    seq_start[0] = ~isnan[0]  # Ensure first value is handled properly
    seq_ids = np.cumsum(seq_start) * ~isnan  # Assign sequence IDs
    unique_ids, counts = np.unique(seq_ids[seq_ids > 0], return_counts=True)

    return np.sum(counts >= 3)  # Count sequences with at least 3 values

def longest_non_nan_sequence(arr):
    """Find the longest contiguous non-NaN sequence in a 1D array and return a mask."""
    isnan = np.isnan(arr)
    if np.all(isnan):  # If all values are NaN, return an all-NaN mask
        return np.full_like(arr, np.nan)

    # Identify groups of consecutive non-NaN values
    mask = np.zeros_like(arr, dtype=bool)
    start, max_start, max_len = None, None, 0

    for i in range(len(arr)):
        if not isnan[i]:  # Start or continue a sequence
            if start is None:
                start = i
        else:  # End of a sequence
            if start is not None:
                seq_len = i - start
                if seq_len > max_len:
                    max_len = seq_len
                    max_start = start
                start = None

    # Handle case where the longest sequence goes until the end
    if start is not None:
        seq_len = len(arr) - start
        if seq_len > max_len:
            max_len = seq_len
            max_start = start

    # Apply the mask to keep only the longest sequence
    if max_start is not None:
        mask[max_start:max_start + max_len] = True

    # Convert False to NaN
    return np.where(mask, arr, np.nan)
