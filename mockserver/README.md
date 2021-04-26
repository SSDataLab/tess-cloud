# Mock MAST TESS FFI server

This directory contains the files needed to run a mock TESS data archive server.
The server will respond to each GET request by returning an identical TICA FFI image.
This is useful for debugging and profiling tess-cut's cutout tool.

## Installation

1. Install Docker.

2. Download a mock FITS image, e.g. from the TICA FFI collection:
```
curl --output image.fits https://archive.stsci.edu/hlsps/tica/s0035/cam4-ccd3/hlsp_tica_tess_ffi_s0035-o1-00149110-cam4-ccd3_tess_v01_img.fits
```

or from the SPOC collection:
```
curl --output image.fits https://archive.stsci.edu/missions/tess/ffi/s0030/2020/270/1-1/tess2020270034913-s0030-1-1-0195-s_ffic.fits
```


3. Build the Docker server image:
```
./build-image.sh
```

4. Run the mock server:
```
./run-image.sh
```

You can now connect to the mock server at http://localhost:8040

## Usage

You can access the example FITS file via any path, e.g.

http://localhost:8040/some/arbitrary/path/test.fits

You can monitor the number of connections and requests via 

http://localhost:8040/nginx_status

You can have `tess-cloud` use this mock server to access images as follows:

```
from tess_cloud import cutout
cutout("Pi Men", sector=1, provider="mock")
```

or alternatively:
```
from tess_cloud import tica
tess_cloud.tica.TICA_MAST_PREFIX = "http://localhost:8040/mock/"
```
