import xarray as xr
import pandas as pd
import numpy as np
import utils
import cftime
from pftnames import pftname
from numpy import empty

'''
    The following functions are all related to CTSM (CLM) post-processing of the grainc variable 
    for the crop functional types (CFTs), which need to match the FAO global statistics crops 
    for crop yield evaluation. Some parts/functions are taken from Sam Rabin's postprocessing scripts,
    from: https://doi.org/10.5194/gmd-16-7253-2023
'''

def plot_global_production(area_grid_m2, grainc_gCm2_grid, Y1,Yn, crop_list,y_ranges):
    ''' 
        plot global mean crop production based on annual CLM grainC model output.
        Area and grainc ds must have same CFT dimension
    '''
    # match years
    area_grid_m2      = area_grid_m2.sel(time=slice(Y1,Yn))
    grainc_gCm2_grid  = grainc_gCm2_grid.sel(time=slice(Y1,Yn))                   
    if len(area_grid_m2.time.values) != len(grainc_gCm2_grid.time.values):
       print('years not matching!')
       return
                         
    # make sure the order of dimensions is equal between datasets
    area_grid_m2 = area_grid_m2.transpose(*grainc_gCm2_grid.dims)

    # set 0 to nan to not mess with averages
    area_grid_m2     = area_grid_m2.where(area_grid_m2>0.0)
    grainc_gCm2_grid = grainc_gCm2_grid.where(grainc_gCm2_grid>0.0)
        
    # check rounding differences and match coords
    if (
        np.allclose(area_grid_m2['lat'], grainc_gCm2_grid['lat']) &
        np.allclose(area_grid_m2['lon'], grainc_gCm2_grid['lon'])
        ):
        area_grid_m2 = area_grid_m2.assign_coords(grainc_gCm2_grid.coords)
        
    production_gC        = grainc_gCm2_grid * area_grid_m2
    productiongC_grouped = group_crops(production_gC,crop_list)
    area_grouped         = group_crops(area_grid_m2, crop_list)

    #set 0 to nan to not mess with averages
    productiongC_grouped = productiongC_grouped.where(productiongC_grouped>0.0)
    area_grouped         = area_grouped.where(area_grouped>0.0)

    # ---- plot global crop timeseries
    n_crops = len(crop_list)
    ncols   = 2
    nrows   = int(np.ceil(n_crops / ncols))
     
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows), sharex=True)
    axes = axes.flatten()  # flatten in case it's a 2D array
    
    # Loop over crops and plot yield time series    
    for i, crop in enumerate(crop_list):
        ax = axes[i]
    
        # Select production and area for the crop
        glob_prod_crop = productiongC_grouped.sel(cfts_grouped=crop).sum(dim=['lat', 'lon'])
        glob_area_crop = area_grouped.sel(cfts_grouped=crop).sum(dim=['lat', 'lon'])
    
        # Compute global yield (tC/ha)
        yield_ts = (glob_prod_crop / glob_area_crop) *0.01 
    
        # Plot
        ax.plot(yield_ts['time'], yield_ts, label=crop, color='tab:blue')
        ax.set_title(crop)
        ax.set_ylabel("Yield (tC/ha)")

        if crop in active_crops:
            crop_index = active_crops.index(crop)
            ymin, ymax = y_ranges[crop_index]
            ax.set_ylim(ymin, ymax)


def group_crops(da, crop_list, Area='', method = 'sum'):
    ''' Check that the crop dimension is 'cft' not 'pft'.
        da       : CLM annual yield dataset (one variable per year)
        crop_list: list of crops that need to be in the output
        area     : only needed for weighted averaging
        method   : group by mean, sum, or weighted mean
    '''
    
    # sums variable per crop type (irrigated, tropical)
    group_crops    = [x.split('_')[-1] for x in da['cft'].values]
    group_series   = pd.Series(group_crops, index=da['cft'].values)
    group_dict     = group_series.groupby(group_series).groups

    grouped = {}

    # group by crop type and selected method
    for group_name, cft_names in group_dict.items():
        subset_group = da.sel(cft=list(cft_names))
        if method == 'sum':
            grouped[group_name] = subset_group.sum(dim='cft')
        elif method == 'mean':
            grouped[group_name] = subset_group.mean(dim='cft')
        elif method == 'weighted_mean':
            subset_area  = Area.sel({dim_crop: list(cft_names)})
            # total area within group
            total_area_group          = subset_area.sum(dim='cft')
            # weighted mean of group
            grouped[group_name] = (subset_group * subset_area ).sum(dim='cft') / total_area_group

    # write output dataset
    grouped_da                 = xr.concat(grouped.values(), dim='cfts_grouped')
    grouped_da['cfts_grouped'] = list(grouped.keys())
    grouped_da                 = grouped_da.sel(cfts_grouped=crop_list)
    
    return grouped_da
    
def reshape_1D_to_2Dgrid(varname, data_in, mask =None):
    '''
        Code to reshape PFT format CLM output to lat/lon format. adapted from CTSM python library: 
        https://github.com/NCAR/ctsm_python_gallery/blob/master/notebooks/PFT-Gridding.ipynb
    '''
    ixy      = data_in.pfts1d_ixy
    jxy      = data_in.pfts1d_jxy
    landfrac = data_in.landfrac
    lat      = data_in.lat
    lon      = data_in.lon
    time     = data_in.time
    vegtype  = data_in.pfts1d_itype_veg
    pfts     = np.array(pftname)

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
    # Only consider crop pfts
    grid_dims = grid_dims.isel(pft=slice(15, 79)) 

    if mask is not None:
        mask = mask.assign_coords({
        'lat': grid_dims.lat,
        'lon': grid_dims.lon,
        'time': grid_dims.time,
        'pft': grid_dims.pft
            })
        grid_dims        = grid_dims.where(mask==1)

    return grid_dims

def production(area_grid_m2, grainc_gCm2_grid, Y1,Yn):
    # area and grainc ds must have same CFT dimension!
    # match years
    area_grid_m2      = area_grid_m2.sel(time=slice(Y1,Yn))
    grainc_gCm2_grid  = grainc_gCm2_grid.sel(time=slice(Y1,Yn))                   
    if len(area_grid_m2.time.values) != len(grainc_gCm2_grid.time.values):
       print('years not matching!')
       return
                         
    # make sure the order of dimensions is equal between datasets
    area_grid_m2 = area_grid_m2.transpose(*grainc_gCm2_grid.dims)

    #set 0 to nan to not mess with averages
    area_grid_m2 = area_grid_m2.where(area_grid_m2>0.0)
    grainc_gCm2_grid = grainc_gCm2_grid.where(grainc_gCm2_grid>0.0)
    
        
    #check rounding differences and match coords
    if (
        (np.allclose(area_grid_m2['lat'], grainc_gCm2_grid['lat'])) & 
         np.allclose(area_grid_m2['lon'], grainc_gCm2_grid['lon'])
        ):
        area_grid_m2 = area_grid_m2.assign_coords(grainc_gCm2_grid.coords)
        
    production_gC = grainc_gCm2_grid * area_grid_m2
    
    return production_gC


def adjust_grainC(da_in, pft1d_itype_veg_str):
    ''' 
        Code from sam Rabin to adjust grainC of different crops accordingly.
        source: https://doi.org/10.5194/gmd-16-7253-2023 --> https://zenodo.org/records/7758123
    '''
    # Parameters from Danica's 2020 paper
    fyield = 0.85 # 85% harvest efficiency (Kucharik & Brye, 2003)
    cgrain = 0.45 # 45% of dry biomass is C (Monfreda et al., 2008)
    
    # Dry matter fraction from Wirsenius (2000) Table A1.II except as noted
    drymatter_fractions = {
        'corn': 0.88,
        'cotton': 0.912, # Table A1.III, "Seed cotton", incl. lint, seed, and "other (ginning waste)"
        'rice': 0.87,
        'soybean': 0.91,
        'sugarcane': 1-0.745, # Irvine, Cane Sugar Handbook, 10th ed., 1977, P. 16. (see sugarcane.py)
        'wheat': 0.88,
    }
    
    # Convert pft1d_itype_veg_str to needed format
    if isinstance(pft1d_itype_veg_str, xr.DataArray):
        pft1d_itype_veg_str = pft1d_itype_veg_str.values
    if not isinstance(pft1d_itype_veg_str[0], str):
        pft1d_itype_veg_int =pft1d_itype_veg_str
        pfts1d_itype_veg_str = [utils.ivt_int2str(x) for x in pft1d_itype_veg_int]
    
    # Create new array with pft as the first dimension. This allows us to use Ellipsis when filling.
    pft_dim = da_in.dims.index('pft')
    wet_tp = np.full(da_in.shape, np.nan)
    wet_tp = np.moveaxis(wet_tp, pft_dim, 0)
    
    # Fill new array, increasing to include water weight
    drymatter_cropList = []
    da_in.load()
    for thisCrop, dm_frac in drymatter_fractions.items():
        drymatter_cropList.append(thisCrop)
        i_thisCrop = [i for i,x in enumerate(pft1d_itype_veg_str) if thisCrop in x]
        
        tmp = da_in.isel(pft=i_thisCrop).values
        if dm_frac != None:
            tmp[np.where(~np.isnan(tmp))] /= dm_frac
        elif np.any(tmp > 0):
            raise RuntimeError(f"You need to get a real dry-matter fraction for {thisCrop}")
        
        # For sugarcane, also account for the fact that soluble solids are only 51% of dry matter. 
        # Also derived from Irvine, Cane Sugar Handbook, 10th ed., 1977, P. 16. (see sugarcane.py)
        if thisCrop == "sugarcane":
            tmp /= 0.51
        
        wet_tp[i_thisCrop, ...] = np.moveaxis(tmp, pft_dim, 0)
    # Move pft dimension (now in 0th position) back to where it should be.
    wet_tp = np.moveaxis(wet_tp, 0, pft_dim)
    
    # Make sure NaN mask is unchanged
    if not np.array_equal(np.isnan(wet_tp), np.isnan(da_in.values)):
        missing_croptypes = [x for x in np.unique(pft1d_itype_veg_str) if x not in drymatter_cropList]
        raise RuntimeError(f'Failed to completely fill wet_tp. Missing crop types: {missing_croptypes}')

    # Save to output DataArray
    da_out = xr.DataArray(data = wet_tp * fyield / cgrain,
                          coords = da_in.coords,
                          attrs = da_in.attrs)
    return da_out


def open_lu_ds(filename, y1, yN, existing_ds, ungrid=True):
    ''' 
        open landuse dataset (add area variable first)
        Code from sam Rabin to adjust grainC of different crops accordingly.
        source: https://zenodo.org/records/7758123 --> cropcal_module.py
    '''
    # Open and trim to years of interest
    dsg = xr.open_dataset(filename).sel(time=slice(y1,yN))
    
    # Assign actual lon/lat coordinates
    dsg = dsg.assign_coords(lon=("lsmlon", existing_ds.lon.values),
                            lat=("lsmlat", existing_ds.lat.values))
    dsg = dsg.swap_dims({"lsmlon": "lon",
                         "lsmlat": "lat"})
    
    if "AREA" in dsg:
        dsg['AREA_CFT'] = dsg.AREA*1e6 * dsg.LANDFRAC_PFT * dsg.PCT_CROP/100 * dsg.PCT_CFT/100
        dsg['AREA_CFT'].attrs = {'units': 'm2'}
        dsg['AREA_CFT'].load()
    else:
        print("Warning: AREA missing from Dataset, so AREA_CFT will not be created")
    
    if not ungrid:
        return dsg
    
    # Un-grid
    query_ilons = [int(x)-1 for x in existing_ds['pfts1d_ixy'].values]
    query_ilats = [int(x)-1 for x in existing_ds['pfts1d_jxy'].values]
    query_ivts = [list(dsg.cft.values).index(x) for x in existing_ds['pfts1d_itype_veg'].values]
    
    ds = xr.Dataset(attrs=dsg.attrs)
    for v in ["AREA", "LANDFRAC_PFT", "PCT_CFT", "PCT_CROP", "AREA_CFT"]:
        if v not in dsg:
            continue
        if 'time' in dsg[v].dims:
            new_coords = existing_ds['GRAINC_TO_FOOD_ANN'].coords
        else:
            new_coords = existing_ds['pft1d_lon'].coords
        if 'cft' in dsg[v].dims:
            ds[v] = dsg[v].isel(lon=xr.DataArray(query_ilons, dims='pft'),
                                lat=xr.DataArray(query_ilats, dims='pft'),
                                cft=xr.DataArray(query_ivts, dims='pft'),
                                drop=True)\
                            .assign_coords(new_coords)
        else:
            ds[v] = dsg[v].isel(lon=xr.DataArray(query_ilons, dims='pft'),
                                lat=xr.DataArray(query_ilats, dims='pft'),
                                drop=True)\
                            .assign_coords(new_coords)
    for v in existing_ds:
        if "pft1d_" in v or "grid1d_" in v:
            ds[v] = existing_ds[v]
    ds['lon'] = dsg['lon']
    ds['lat'] = dsg['lat']
    
    # Which crops are irrigated?
    is_irrigated = np.full_like(ds['pfts1d_itype_veg'], False)
    for vegtype_str in np.unique(ds['pfts1d_itype_veg_str'].values):
        if "irrigated" not in vegtype_str:
            continue
        vegtype_int = utils.ivt_str2int(vegtype_str)
        is_this_vegtype = np.where(ds['pfts1d_itype_veg'].values == vegtype_int)[0]
        is_irrigated[is_this_vegtype] = True
    ["irrigated" in x for x in ds['pfts1d_itype_veg_str'].values]
    ds['IRRIGATED'] = xr.DataArray(data=is_irrigated,
                                   coords=ds['pfts1d_itype_veg_str'].coords,
                                   attrs={'long_name': 'Is pft irrigated?'})
    
    # How much area is irrigated?
    ds['IRRIGATED_AREA_CFT'] = ds['IRRIGATED'] * ds['AREA_CFT']
    ds['IRRIGATED_AREA_CFT'].attrs = {'long name': 'CFT area (irrigated types only)',
                                      'units': 'm^2'}
    ds['IRRIGATED_AREA_GRID'] = (ds['IRRIGATED_AREA_CFT']
                                 .groupby(ds['pft1d_gi'])
                                 .sum()
                                 .rename({'pft1d_gi': 'gridcell'}))
    ds['IRRIGATED_AREA_GRID'].attrs = {'long name': 'Irrigated area in gridcell',
                                      'units': 'm^2'}
    
    return ds
    
def detrend_yield(ds_yield_years, T_dim='year', fit_degree =1):
    ''' 
    detrend yield based on linear trend (if polyfit degree=1)
    '''
    # Fit a linear trend (degree=1)
    poly_coeffs = ds_yield_years.polyfit(dim=T_dim, deg=fit_degree)

    # Compute the trend line
    valid_mask = ds_yield_years.notnull()
    trend = xr.polyval(ds_yield_years[T_dim], poly_coeffs.polyfit_coefficients).where(valid_mask)
    
    # Detrend by subtracting the trend
    y_detrend = ds_yield_years - trend
    return y_detrend

def detrend_yield_rolling(ds_yield_years, T_dim='year', window=5):
    ''' 
    detrend yield based on rolling mean (default = 5 years)
    '''
    rolling_mean = ds_yield_years.rolling({T_dim: window}, center=True).mean()
    rolling_mean = rolling_mean.interpolate_na(dim=T_dim, method="linear")  # fill edge NaNs

    # Subtract rolling mean to get "detrended" anomalies
    y_detrend = ds_yield_years - rolling_mean
    
    return y_detrend

def standardized_yield_anomalies(ds_yield_years, T_dim='year', detrend_method ='rolling_mean'):
    '''
    Compute standardized yield anomalies.
    
    Parameters:
    - ds_yield_years: xarray DataArray of yield with time dimension T_dim
    - T_dim: str, name of the time dimension (default 'year')
    
    Returns:
    - xarray DataArray of standardized yield anomalies
    '''
    # Step 1: Detrend (optional, can use your detrend_yield function)
    if detrend_method == 'rolling_mean':
        ds_detrend = detrend_yield_rolling(ds_yield_years, T_dim=T_dim)
    elif detrend_method == 'linear':
        ds_detrend = detrend_yield(ds_yield_years, T_dim=T_dim)

    # Step 2: Compute mean and std along time
    mean_yield = ds_detrend.mean(dim=T_dim)
    std_yield  = ds_detrend.std(dim=T_dim)

    # Step 3: Standardize
    standardized_anom = (ds_detrend - mean_yield) / std_yield

    return standardized_anom
    

def global_production_yield(area_grid_m2, grainc_gCm2_grid, Y1,Yn, crop_list,y_ranges=0, plot='off'):
    '''
    ***** Built on bit's and pieces from Sam's code (2023: https://doi.org/10.5194/gmd-16-7253-2023) *****
    groups cft's together that are grouped under the same crop 
    (e.g. corn = temperate corn, irrigated temperate corn, tropical corn, irrigated tropical corn)
    1. calculates production for each cft
    2. sum for each year and grid the total production for each crop group
    3. sum the area for each group for each year and grid
    area_grid_m2     : annual cft area in m2
    grainc_gCm2_grid : annual grainc data in g/m2 !note!! this is already corrected for to yield grainc!
    Y1               : start year
    Y2               : end year
    crop_list        : groups of crops which should be outputted
    plot             : if plot == 'off' a dataset will be created with production, area and yield per crop group
                       if plot == 'on' global production plots will be outputted
    area and grainc ds must have same CFT dimension!
    '''
    # match years
    area_grid_m2      = area_grid_m2.sel(time=slice(Y1,Yn))
    grainc_gCm2_grid  = grainc_gCm2_grid.sel(time=slice(Y1,Yn))                   
    if len(area_grid_m2.time.values) != len(grainc_gCm2_grid.time.values):
       print('years not matching!')
       return
                         
    # make sure the order of dimensions is equal between datasets
    area_grid_m2 = area_grid_m2.transpose(*grainc_gCm2_grid.dims)

    #set 0 to nan to not mess with averages
    area_grid_m2 = area_grid_m2.where(area_grid_m2>0.0)
    grainc_gCm2_grid = grainc_gCm2_grid.where(grainc_gCm2_grid>0.0)
        
    #check rounding differences and match coords
    if (
        (np.allclose(area_grid_m2['lat'], grainc_gCm2_grid['lat'])) &
         np.allclose(area_grid_m2['lon'], grainc_gCm2_grid['lon'])
       ):
        area_grid_m2 = area_grid_m2.assign_coords(grainc_gCm2_grid.coords)
        
    production_gC = grainc_gCm2_grid * area_grid_m2
        
    productiongC_grouped = group_crops(production_gC,crop_list, method='sum')
    area_grouped         = group_crops(area_grid_m2, crop_list, method='sum')

    #set 0 to nan to not mess with averages
    productiongC_grouped = productiongC_grouped.where(productiongC_grouped>0.0)
    area_grouped         = area_grouped.where(area_grouped>0.0)
        
    if plot == 'off':
        #return dataset with yield and production grouped by crop type (rainfed, irrigated and trop merged)
        yield_crops = (productiongC_grouped / area_grouped) * 0.01
        ds_out = xr.Dataset(
            data_vars={
                "production_crop": productiongC_grouped,
                "area_crop": area_grouped,
                "yield_crop": yield_crops,
            },
            coords=grainc_gCm2_grid.coords
        )
        
        return ds_out
        
    elif plot == 'on':
        # ---- plot global crop timeseries
        # Loop over crops and plot yield time series
        n_crops = len(crop_list)
        
        # Set up subplots (auto-arranged in a grid)
        ncols = 2 
        nrows = int(np.ceil(n_crops / ncols))
        
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows), sharex=True)
        axes = axes.flatten()  # flatten in case it's a 2D array
        
        for i, crop in enumerate(crop_list):
            ax = axes[i]
        
            # Select production and area for the crop
            glob_prod_crop = productiongC_grouped.sel(cfts_grouped=crop).sum(dim=['lat', 'lon'])
            glob_area_crop = area_grouped.sel(cfts_grouped=crop).sum(dim=['lat', 'lon'])
        
            # Compute global yield (tC/ha)
            yield_ts = (glob_prod_crop / glob_area_crop) *0.01 
        
            # Plot
            ax.plot(yield_ts['time'], yield_ts, label=crop, color='tab:blue')
            ax.set_title(crop)
            ax.set_ylabel("Yield (tC/ha)")
    
            if (crop in crop_list) & (y_ranges!= 0):
                crop_index = crop_list.index(crop)
                ymin, ymax = y_ranges[crop_index]
                ax.set_ylim(ymin, ymax)