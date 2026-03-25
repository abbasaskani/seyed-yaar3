# CHANGES APPLIED — lean optimized backend

## Main runtime defaults
- skipjack-only by default
- past-days = 1
- future-days = 5
- step-hours = 12
- grid = 160x160

## Performance optimizations
- Copernicus depth describe cached across the full run
- existing subset NetCDF files are reused instead of re-downloading
- no extra online wind fetch by default; local ERA5 or cheap proxy only
- heavy diagnostic/extended layer writes are off by default
- only the original UI-facing core outputs are written by default
- front persistence and lag features reuse in-memory time caches
- netCDF variables are read via netCDF4 instead of repeated rasterio opens

## Scientific features kept active
- BOA SST front
- BOA logCHL front
- SSH front
- 3-day front persistence
- EKE
- Okubo-Weiss
- eddy-edge distance
- MLD / O2 / SSS / vertical_access
- CHL 3d mean
- simple surface wind in ops

## Present in code but off by default
- 7-day front persistence
- CHL 7d / CHL anomaly / NPP anomaly
- vorticity / strain output writes
- extended diagnostic layer exports
