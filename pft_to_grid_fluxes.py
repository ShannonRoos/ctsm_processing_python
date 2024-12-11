import numpy as np
from numpy import empty
import warnings
warnings.filterwarnings('ignore')
import xarray as xr
import pftnames
import utils

''' Python file to convert PFT formatted CLM output to gridded output using area maps for CFT. Use landuse map to get
    the correct crop area (otherwise inactive crops are lumped together with active crop areas). 
    Adapted from:  https://github.com/NCAR/ctsm_python_gallery/blob/master/notebooks/PFT-Gridding.ipynb'''


'''--------------------- input by user ---------------------'''
BASEDIR         = '/dodrio/scratch/projects/2022_200/project_output/bclimate/sderoos/apps/CTSM_dev/ctsm_master'
CASE            = 'ihist.e52.IHistClm50BgcCrop.f19_g17.control'
run             = 'h3'     #'h3', 'h4'
vars            = ['GPP','TLAI','GRAINC_TO_FOOD', 'GRAINC_TO_FOOD_ANN']
varname         = vars[0]

'''--------------------- assign directories -----------------'''
case            = f'{BASEDIR}/cases/{CASE}'
model_out       = f'{BASEDIR}/output_processed/{CASE}/'
path_out        = f'{BASEDIR}/output_processed/{CASE}/'
apply_cropmask  = False     #True

'''--------------------- find matching landuse map ---------------------'''
if CASE.split('.')[1] == 'e22' and CASE.split('.')[3] == 'f19_g17':
    lucdat = xr.open_dataset(
        '/dodrio/scratch/projects/2022_200/project_input/cesm/inputdata/lnd/clm2/surfdata_map/'
        'landuse.timeseries_1.9x2.5_hist_78pfts_CMIP6_simyr1850-2015_c170824.nc')
elif CASE.split('.')[1] == 'e52' and CASE.split('.')[3] == 'f19_g17':
    lucdat = xr.open_dataset(
        '/dodrio/scratch/projects/2022_200/project_input/cesm/inputdata/lnd/clm2/surfdata_esmf/ctsm5.2.0/'
        'landuse.timeseries_1.9x2.5_SSP2-4.5_1850-2100_78pfts_c240216.nc')
else:
    print('cannot find surface data map, check CASE name model version and resolution')

'''--------------------- define temporal resolution ---------------------'''
if run == 'h0' or run == 'h2':
    tsteps          = 'month'
    ntimes          = np.arange(1, 13, 1)
    timename        = 'nmonth'
elif run == 'h1' or run == 'h3':
    tsteps          = 'days'
    ntimes          = np.arange(1, 366, 1)
    timename        = 'ndays'
elif run == 'h4':
    tsteps          = 'years'
    ntimes          = np.arange(1,2, 1)
    timename        = 'nyears'
else:
    print('ERROR: no valid CLM input dataset')


'''---------------------  open clm dataset ---------------------'''
fname           = f'{model_out}{CASE}.clm2.{run}.nc'

# daily values are end of day --> represent day before
data1           = utils.time_set_mid(xr.open_dataset(fname, decode_times=True), 'time')

area            = data1.area
landfrac        = data1.landfrac
lat             = data1.lat
lon             = data1.lon
time            = data1.time

ixy             = data1.pfts1d_ixy
jxy             = data1.pfts1d_jxy
coltype         = data1.pfts1d_itype_col
vegtype         = data1.pfts1d_itype_veg
cellwtd         = data1.pfts1d_wtgcell

var             = data1[varname]

# get size of dimensions
nlat            = len(lat.values)
nlon            = len(lon.values)
nvegtype        = len(vegtype.values)
ntim            = len(time.values)
npft            = (np.max(vegtype))
pfts            = np.array(pftnames.pftname)
npft            = npft.astype(int) + 1
pftlist         = np.arange(0, (npft.values) +1, 1)

# define new empty dataset with correct dimensions
gridded         = empty([ntim,npft.values,nlat,nlon])
print('here')

# Fill in empty dataset with CLM output
gridded[:, vegtype.values.astype(int), jxy.values.astype(int) - 1, ixy.values.astype(int) - 1] = var.values
grid_dims       = xr.DataArray(gridded[1:,:,:], dims=("time","pft","lat","lon"))
grid_dims       = grid_dims.assign_coords(time=data1.time,pft=pfts,lat=lat.values,lon=lon.values)
grid_dims.name  = var
grid_dims       = grid_dims.where(data1.landfrac==True)
years           = np.unique(grid_dims.indexes['time'].year)
nyears          = len(years)

if apply_cropmask == True:
    # set annual transient cropmap as mask for CFTs
    pctcft          = lucdat.PCT_CFT
    cropmask        = pctcft.where(pctcft > 0.0)
    # not interested in percentages, so we need a binary map
    binary_mask     = xr.where((cropmask.notnull()), 1, 0)
    binary_mask     = binary_mask.rename({'time': 'year', 'cft': 'pft', 'lsmlat': 'lat', 'lsmlon': 'lon'})
    yr_start        = np.where(binary_mask.year == years[0])[0][0]
    yr_end          = np.where(binary_mask.year == years[-1])[0][0]
    binary_mask     = binary_mask[yr_start:yr_end+1,:,:,:]


    # reshaping the time dimension to year, ndays (to filter data by annual masks)
    reshaped        = grid_dims.values.reshape(nyears, len(ntimes), *grid_dims.values.shape[1:])
    grid_reshape    = xr.DataArray(reshaped, dims=("year", "doy", "pft", "lat", "lon"))
    grid_renamed    = grid_reshape.assign_coords(year=years, doy=ntimes, pft=pftnames.pftname, lat=lat.values,
                                              lon=lon.values)

    # add dimension day to mask to match clm grid
    expanded_mask   = binary_mask.expand_dims(doy=grid_reshape.doy).transpose("year", "doy", "pft", "lat", "lon")

    # subset clm dataset to match mask
    sub_clm             = grid_renamed.isel(pft=slice(15, 79))
    binary_mask['pft']  = sub_clm.pft
    binary_mask['year'] = sub_clm.year
    # mask data
    sub_clm = sub_clm.where(binary_mask)


    # convert back to time,pft,lat,lon dimensions
    sub_clm = sub_clm.stack(time=("year", "doy"))
    sub_clm = sub_clm.assign_coords(time=("time", data1.time.values))
    sub_clm = sub_clm.transpose("time", ...)

    # change run name for output filename
    run                 = run + '_cropmask'

else:
    sub_clm             = grid_dims


# important --> give a name to the var dimensions otherwise it will give an error when saving as xarray dataset
sub_clm.name = varname
sub_clm.to_netcdf(f'{path_out}{CASE}.{run}_{varname}_pft.nc')