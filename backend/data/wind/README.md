Place optional local ERA5 daily NetCDF files here if you want real 10m wind without any extra online fetch in the lean pipeline.

Expected filename pattern:
  era5_YYYYMMDD.nc

Expected variables:
  u10
  v10

If the file is missing, the runtime falls back to a cheap wave/current surface proxy so the pipeline stays fast.
