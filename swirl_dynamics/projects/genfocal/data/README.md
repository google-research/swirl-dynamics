# Data for GenFocal
This directory contains datasets related to GenFocal. These datasets include
those necessary to reproduce the results of the paper, as well as additional
datasets that may be useful for future work. Colabs in the colabs/ directory
demonstrate how to load and visualize the datasets using Google Colab.

The data directory is organized as follows:
```
data/
├-- colabs/
├-- era5/
    ├-- era5_240x121_lonlat_1980-2020_10_vars.zarr/
    ├-- era5_240x121_lonlat_2010-2011_10_vars.zarr/
├-- inference
    ├-- conus/
        ├-- debiased_100members_jja10s_8samples_xm153985229_member_chunks.zarr/
        ├-- debiased_100members_jja10s_8samples_xm153999662_pixel_chunks.zarr/
    ├-- nao/
        ├-- debiased_100members_aso10s_8samples_xm155030721_member_chunks.zarr/
├-- lens_debiased
    ├-- xm_151818801_1_100_members_1960_2100_gen_154117335.zarr/
├-- lens2
    ├-- lens2_240x121_lonlat_1960-2020_10_vars_4_train_members.zarr/
    ├-- lens2_240x121_lonlat_1960-2100_10_vars_100_members.zarr/
```
## ERA5
This directory contains downsampled ERA5 data matching the grid used by
[LENS2](https://www.cesm.ucar.edu/community-projects/lens2).
Data is available for the years 1980-2020 and 2010-2011, with daily samples.
Each dataset is a zarr volume, with the following surface variables:
 - temperature 2m above ground
 - mean sea level pressure
 and the following variables at 1000, 850, 500 and 250 hPa levels:
 - geopotential
 - specific humidity
 - u component of wind
 - v component of wind

Contains modified Copernicus Climate Change Service information (2025)

## Inference
This directory contains examples of the debiased and downscaled forecasts
produced by GenFocal
### CONUS
This directory contains GenFocal forecasts over the continental United States
(CONUS) for the months of June, July, and August for 2010-2019. Each dataset is
a zarr file containing these variables:
 - wind speed at 10 meters
 - temperature at 2 meters
 - mean sea level pressure
 - near-surface specific humidity
One copy of the dataset is chunked by grid pixel, facilitating visualization of
time series. The other
copy is chunked by ensemble member to facilitate spatial visualizations.
### nao
This directory contains GenFocal forecasts over the North Atlantic hurricane
basin (NAO) for the months of June, July, and August for 2010-2019. The dataset
is a zarr file containing these variables:
 - wind speed at 10 meters
 - temperature at 2 meters
 - mean sea level pressure
 - near-surface specific humidity
 - geopotential at 200 and 500 hPa

## LENS_DEBIASED
This directory contains debiased
[LENS2](https://www.cesm.ucar.edu/community-projects/lens2) data. There are
100 ensemble members, covering the period of 1960-2100, with daily intervals.
The surface variables are:
 - wind speed at 10 meters
 - temperature at 2 meters
 - mean sea level pressure
 - near-surface specific humidity

 The upper air variables are:
  - U/V winds at 200 and 850 hPa
  - geopotential at 200 and 500 hPa

## LENS2
This directory contains two
[LENS2](https://www.cesm.ucar.edu/community-projects/lens2) datasets. Both
datasets have data at daily intervals for the period of 1960-2020
The surface variables are:
 - wind speed at 10 meters
 - temperature at 2 meters
 - mean sea level pressure
 - near-surface specific humidity

 The upper air variables are:
  - U/V winds at 200 and 850 hPa
  - geopotential at 200 and 500 hPa

### LENS2_240x121_lonlat_1960-2020_10_vars_4_train_members.zarr
This dataset contains 4 members of the LENS2 dataset.

### LENS2_240x121_lonlat_1960-2100_10_vars_100_members.zarr
This dataset contains all 100 members of the LENS2 dataset.

## License
The datasets are licensed using the CC-BY-4.0 license.
The colab notebooks are licensed using the Apache 2.0 license.