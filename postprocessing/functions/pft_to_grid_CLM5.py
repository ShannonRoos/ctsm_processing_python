import numpy as np
from numpy import empty
import xarray as xr
import pftnames
import cftime
import utils
import function_reshape_pft_CLM as reshapeCLM


''' ------------------------------------------------------------------------------------------
    Python file to convert PFT formatted CLM output to gridded output using area maps for CFT. 
    Use landuse map to get the correct crop area (otherwise inactive crops are lumped together 
    with active crop areas). Adapted from:  
    https://github.com/NCAR/ctsm_python_gallery/blob/master/notebooks/PFT-Gridding.ipynb
    ------------------------------------------------------------------------------------------ '''


'''--------------------- input by user '''
lucdat          = xr.open_dataset('inputdata/lnd/clm2/surfdata_esmf/ctsm5.2.0/'
                                  'landuse.timeseries_1.9x2.5_SSP2-4.5_1850-2100_78pfts_c240216.nc')
expname         = 'exp0'
BASEDIR         = f''
CASE            = f'{BASEDIR}ihist.e52.IHistClm50BgcCrop.f19_g17.3.HSF_{expname}/'
run             = 'h3'
varnames        = ['GPP','ELAI', 'TVDAY','TVMAX']
years           = np.arange(1980,2015)


if run == 'h1' or run =='h3':
    for y in years:
        fout  = f'{CASE}ELAI_h3_test_{y}.nc'
        fname = f'{CASE}ihist.e52.IHistClm50BgcCrop.f19_g17.3.HSF_{expname}.clm2.{run}.{y}-01-01-00000.nc'

        '''--------------------- load pft file '''
        data1 = utils.time_set_mid(xr.open_dataset(fname, decode_times=True), 'time')
        time  = data1.time
        years = np.unique(time.dt.year)

        '''--------------------- generate crop mask based on land cover pct map '''
        pctcft   = lucdat.PCT_CFT
        cropmask = pctcft.where(pctcft > 0.0)
        mask     = xr.where((cropmask.notnull()), 1, np.nan)
        mask     = mask.rename({'time': 'year', 'cft': 'pft', 'lsmlat': 'lat', 'lsmlon': 'lon'})
        yr_main  = np.where(mask.year == years[-1])[0][0]
        yr_mask  = mask[yr_main, :, :, :]
        mask_pft = yr_mask.expand_dims(time=time).copy(deep=True)

        '''--------------------- reshape grid and save output '''
        # Create a dataset with all variables
        reshaped_vars = {
            var: reshapeCLM.reshape_1D_to_2Dgrid(var, data1, mask_pft) for var in varnames
        }

        # Add the fixed variables
        reshaped_vars.update({
            "gridweights": reshapeCLM.reshape_1D_to_2Dgrid('pfts1d_wtgcell', data1),
            "grid_col": reshapeCLM.reshape_1D_to_2Dgrid('pfts1d_itype_col', data1)
        })

        # Create the merged dataset
        ds_merged = xr.Dataset(reshaped_vars)

        # add attributes
        for var in ds_merged.data_vars:
            if var in data1:
                ds_merged[var].attrs = data1[var].attrs.copy()

        ds_merged.to_netcdf(fout)

if run == 'h4':
    '''h4 output files are annual yield output files in a single file'''
    '''--------------------- load pft file '''
    fname = f'{CASE}ihist.e52.IHistClm50BgcCrop.f19_g17.3.HSF_exp0.clm2.h4.1980-01-01-00000.nc'
    fout  = f'{CASE}GRAINC_TO_FOOD_{years[0]}_{years[-1]}.nc'
    data1 = xr.open_dataset(fname, decode_times=True)
    time  = data1.time
    years = np.unique(time.dt.year)

    '''--------------------- generate crop mask based on crop cover percentage '''
    pctcft   = lucdat.PCT_CFT
    cropmask = pctcft.where(pctcft > 0.0)
    mask     = xr.where((cropmask.notnull()), 1, np.nan)
    mask     = mask.rename({ 'cft': 'pft', 'lsmlat': 'lat', 'lsmlon': 'lon'})
    yr_start = np.where(mask.time == years[0])[0][0]
    yr_end   = np.where(mask.time == years[-1]+1)[0][0]
    mask_pft = mask[yr_start:yr_end , :, :, :]

    '''--------------------- reshape grid and save output '''

    ds_merged = xr.Dataset({
        "GRAINC_TO_FOOD_ANN": reshapeCLM.reshape_1D_to_2Dgrid('GRAINC_TO_FOOD_ANN', data1, mask_pft),
        "gridweights": reshapeCLM.reshape_1D_to_2Dgrid('pfts1d_wtgcell', data1),
        "grid_col": reshapeCLM.reshape_1D_to_2Dgrid('pfts1d_itype_col', data1)
    })

    # add attribute information
    for var in ds_merged.data_vars:
        if var in data1:
            ds_merged[var].attrs = data1[var].attrs.copy()

    ds_merged.to_netcdf(fout)


