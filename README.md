# ctsm_processing_python
python code related to CTSM (v5.2)

contains:
- pft_to_grid_CLM5.ipynb      : convert CLM output in PFT to gridded datasets, uses:
- function_reshape_pft_CLM.py : contains core function to reshape CLM pft format to lat-lon grid

- crop_weights.py: functions related to rescaling CLM output consdering weighted average of crop percentages and converting GRAIN_C_TO_FOOD to yield in t/ha

# functions organized per category:
- LAI_functions.py : functions related to calculation of LAI slope
- HS_functions.py  : functions related to definition of heatwave
- grainc_global.py : CLM postprocessing functions for annual yield

