# ctsm_processing_python
python code related to CTSM (v5.2)
Most of the code is postprocessing of CLM5 data or evaluation data (crop yield /LAI)

### postprocessing
contains:

#### functions organized per category:
- LAI_functions.py : functions related to calculation of LAI slope
- HS_functions.py  : functions related to definition of heatwave
- GRAINC_functions.py : CLM postprocessing functions for annual yield
#### notebooks to execute evaluation:
- mask_CGLS_LAI.ipynb:


### remap_lai_agriculture
contains python and bash scripts to harmonize LAI datasets from GLASS and CGLS, extract agricultural information and aggregate to CLM5 desired resolution.
- processing_bash: simple remapping scripts using CDO
- mask_CGLS_LAI.ipynb: 
- mask_GLASS_LAI.ipynb:

See workflow in image below

![Workflow overview](assets/flowchart.jpg)
