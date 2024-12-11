import numpy as np
import xarray as xr


lucdatclm22_2deg = ('/dodrio/scratch/projects/2022_200/project_input/cesm/inputdata/lnd/clm2/surfdata_map/'
                    'landuse.timeseries_1.9x2.5_hist_78pfts_CMIP6_simyr1850-2015_c170824.nc')
lucdatctsm52_2deg = ('/dodrio/scratch/projects/2022_200/project_input/cesm/inputdata/lnd/clm2/surfdata_esmf/ctsm5.2.0/'
                     'landuse.timeseries_1.9x2.5_SSP2-4.5_1850-2100_78pfts_c240216.nc')


def GRAINC_to_yield(ds_clm, crop_clm, run):
    # need to sum daily GRAINC_TO_FOOD (g/m2/s) and convert to annual yield in t/ha
    ds_clm = ds_clm['GRAINC_TO_FOOD']
    '''--------------------- convert clm output to annual yield in t/ha -----------------'''
    gm2_to_tha = 0.01
    seconds_in_days = 86400
    grainC_dryweight = 0.45
    harvest_efficiency = 0.85
    dry_to_wet = {'corn': 0.88, 'cotton': 0.912, 'rice': 0.87, 'soybean': 0.91, 'sugarcane': 0.255, 'wheat': 0.88}
    # calc clm yield
    if run == 'h3':
        # 1. unit is gC/m2/sec --> sum daily fluxes of GRAINC_TO_FOOD and convert to g/m2/d
        ds_clm_ann = ds_clm.groupby("time.year").sum(dim="time") * seconds_in_days
        # 1. apply corrections to get annual yiel in t/ha
        grain_yield_corrected = ((ds_clm_ann * harvest_efficiency / grainC_dryweight) * gm2_to_tha) / dry_to_wet[crop_clm]
    elif run == 'h4':
        # unit is gC/m2 (already converted to annual GRAINC_TO_FOOD; skip step 1)
        grain_yield_corrected = ((ds_clm * harvest_efficiency / grainC_dryweight) * gm2_to_tha) / dry_to_wet[crop_clm]

    clm_yield_all = grain_yield_corrected
    return(clm_yield_all)


def read_wcrop_clm(f, crop, var,  management):
    '''Get the weighted average of variable respective to specific crop type presence.
       This function takes into account the fractional weights of either irrigated/rainfed/irrigated+rainfed'''
    
    #extract model case and res (specific for my model setup)
    versionclm = f.split('/')[-1].split('.')[1]
    res        = f.split('/')[-1].split('.')[3]

    '''--------------------- find matching landuse map ---------------------'''
    if versionclm == 'e22'and res == 'f19_g17':
        lum = xr.open_dataset(lucdatclm22_2deg)
    elif versionclm == 'e52' and res == 'f19_g17':
        lum = xr.open_dataset(lucdatctsm52_2deg)
    else:
        print('cannot find surface data map, check model version and resolution')

    # read in clm data
    ds_clm    = xr.open_dataset(f)
    crops     = ds_clm['pft'].values
    indices   = [index for index, element in enumerate(crops) if crop in element]
    ind_names = crops[indices]
    ds_var    = ds_clm[var][:, indices, :, :]

    # reshaping the time dimension to year, ndays (to filter data by annual masks)
    time               = ds_clm['time'].values
    years              = np.unique(ds_clm.indexes['time'].year)
    ntimes             = int(len(ds_clm['time']) / len(years))
    times              = np.arange(1,ntimes+1,1)
    reshaped           = ds_var.values.reshape(len(years), ntimes, *ds_var.values.shape[1:])
    grid_reshape       = xr.DataArray(reshaped, dims=("year", "doy", "cft", "lat", "lon"))
    ds_var_reshaped    = grid_reshape.assign_coords(year=years, doy=times, cft=ds_var['pft'].values,
                                                    lat=ds_var.lat.values,lon=ds_var.lon.values)

    # replace land use map (lum) cft numbers with crop names
    # replace land use map (lum) cft numbers with crop names
    lum        = lum.rename({'time': 'year', 'lsmlat': 'lat', 'lsmlon': 'lon'})
    lum        = lum.assign_coords({"cft": ds_clm['pft'].values, 'lon':ds_var.lon.values, 'lat':ds_var.lat.values})
    lum        = lum.sel(year=ds_var_reshaped.year)
    lum        = lum.expand_dims(doy=times)
    pct_crop   = lum['PCT_CFT'].transpose("year", "doy", "cft", "lat", "lon")
    pct_crop   = pct_crop[:, :,indices,:,:]

    # select crops based on management (irrigated / rainfed)
    if management == 'rainfed':
        # 1 --- select all pfts related to rainfed crop
        idx_rf = [index for index, element in enumerate(ind_names) if not 'irrigated' in element]
        crop_clm = ds_var_reshaped[:,:,idx_rf,:,:]
        pct_crop = pct_crop[:,:,idx_rf,:,:]
        print('rainfed only')
    elif management == 'irrigated':
        # 2 --- select irrigated crop pfts
        idx_irr = [index for index, element in enumerate(ind_names) if 'irrigated' in element]
        crop_clm = ds_var_reshaped[:,:, idx_irr, :, :]
        pct_crop = pct_crop[:,:, idx_irr, :, :]
        print('irrigated only')
    elif management == 'all' or management == '':
        crop_clm = ds_var_reshaped
        pct_crop = pct_crop
        print('all manamgement')

    # get weighted average of crop
    pct_cft_normalized = pct_crop / pct_crop.sum(dim='cft')
    ds_var_rescaled = (crop_clm * pct_cft_normalized).sum(dim='cft')

    if var == 'GRAINC_TO_FOOD':
        # calc clm yield
        gm2_to_tha = 0.01
        seconds_in_days = 86400
        grainC_dryweight = 0.45
        harvest_efficiency = 0.85
        dry_to_wet         = {'corn': 0.88, 'cotton': 0.912, 'rice': 0.87,
                              'soybean': 0.91, 'sugarcane': 0.255, 'wheat': 0.88}
        # unit is gC/m2/sec
        #sum GRAINC_TO_FOOD for each year
        ds_var_ann = ds_var_rescaled.sum(dim="doy") * seconds_in_days
        # Get annual yield using CLM specific calculations
        ds_var_out  = ((ds_var_ann * harvest_efficiency / grainC_dryweight) * gm2_to_tha) / dry_to_wet[crop]

    elif var == 'GPP' or var =='TLAI' or var == 'ELAI':
        # convert year + doy back to single time dimension
        #ds_var_out         = ds_var_rescaled.assign_coords(time=("time", time)).swap_dims({"year": "time"}).drop("doy")
        ds_var_rescaled = ds_var_rescaled.stack(time=("year", "doy"))
        ds_var_rescaled = ds_var_rescaled.drop_vars(['time', 'year', 'doy'])
        ds_var_rescaled = ds_var_rescaled.assign_coords(time=("time", time.values))         #   should be midday need to add +12hrs --> fixed in pft_to_fluxes
        ds_var_out = ds_var_rescaled.transpose("time", ...)

    return ds_var_out


def read_wcrops_all_clm(f, management, var):
    ''' get weighted average of different crops in pixel'''
    '''this function takes into account the fractional weights of either irrigated/rainfed/irrigated+rainfed'''

    #extract Case model and res
    versionclm  = f.split('/')[-1].split('.')[1]
    res         = f.split('/')[-1].split('.')[3]

    '''--------------------- find matching landuse map ---------------------'''
    if versionclm == 'e22'and res == 'f19_g17':
        lum = xr.open_dataset(lucdatclm22_2deg)
    elif versionclm == 'e52' and res == 'f19_g17':
        lum = xr.open_dataset(lucdatctsm52_2deg)
    else:
        print('cannot find surface data map, check model version and resolution')

    # read in clm data
    ds_clm              = xr.open_dataset(f)
    crops               = ds_clm['pft'][:].values
    ds_var              = ds_clm[var][:, :, :, :]

    # reshaping the time dimension to year, ndays (to filter data by annual masks)
    time               = ds_clm['time'].values
    years              = np.unique(ds_clm.indexes['time'].year)
    ntimes             = int(len(ds_clm['time']) / len(years))
    times              = np.arange(1,ntimes+1,1)
    reshaped           = ds_var.values.reshape(len(years), ntimes, *ds_var.values.shape[1:])
    grid_reshape       = xr.DataArray(reshaped, dims=("year", "doy", "cft", "lat", "lon"))
    ds_var_reshaped    = grid_reshape.assign_coords(year=years, doy=times, cft=ds_var['pft'].values,
                                                    lat=ds_var.lat.values, lon=ds_var.lon.values)

    # read in land use map (lum)
    # replace lum cft numbers with crop names
    lum        = lum.rename({'time': 'year', 'lsmlat': 'lat', 'lsmlon': 'lon'})
    lum        = lum.assign_coords({"cft": ds_clm['pft'].values, 'lon':lum['LONGXY'][1,:].values, 'lat':lum['LATIXY'][:,1].values})
    lum        = lum.sel(year=ds_var_reshaped.year)
    lum        = lum.expand_dims(doy=times)
    pct_crop   = lum['PCT_CFT'].transpose("year", "doy", "cft", "lat", "lon")
    pct_crop   = pct_crop[:, :,indices,:,:]


    #exclude first two 'generic C3 crops' -->[2:])
    if management == 'rainfed':
        # 1 --- select all pfts related to rainfed crop
        idx_rf      = [index for index, element in enumerate(crops) if not 'irrigated' in element]
        crop_clm    = ds_var_reshaped[:,:,idx_rf[1:],:,:]
        pct_crop    = pct_crop[:,:, idx_rf[1:], :, :]
    elif management == 'irrigated':
        # 2 --- select irrigated crop pfts
        idx_irr     = [index for index, element in enumerate(crops) if 'irrigated' in element]
        crop_clm    = ds_var_reshaped[:, :, idx_irr[1:], :, :]
        pct_crop    = pct_crop[:, :,idx_irr[1:], :, :]
    elif management == 'all' or management == '':
        crop_clm    = ds_var_reshaped[:, :,2:, :, :]
        pct_crop    = pct_crop[:,:, 2:, :, :]

    #calculate weighted crop vars
    pct_cft_normalized = pct_crop / pct_crop.sum(dim='cft')
    ds_var_rescaled    = (crop_clm * pct_cft_normalized).sum(dim='cft')

    #reshape back to time,lat,lon
    #ds_var_out         = ds_var_rescaled.assign_coords(time=("time", time)).swap_dims({"year": "time"}).drop("doy")
    ds_var_rescaled = ds_var_rescaled.stack(time=("year", "doy"))
    ds_var_rescaled = ds_var_rescaled.drop_vars(['time', 'year', 'doy'])
    ds_var_rescaled = ds_var_rescaled.assign_coords(time=("time", time))         #   should be midday need to add +12hrs --> fixed in pft_to_fluxes
    ds_var_out = ds_var_rescaled.transpose("time", ...)

    return ds_var_out
