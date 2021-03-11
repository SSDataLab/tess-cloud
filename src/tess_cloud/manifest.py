# TODO: Feb 27: MANIFEST on S3 doesn't appear to have been updated since sector 26.  Caution!
from functools import lru_cache
import io
import os

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import NoCredentialsError

import diskcache
import pandas as pd

from . import TESS_S3_BUCKET


__all__ = ["list_images", "get_s3_uri"]

# Setup the disk cache
CACHEDIR = os.path.join(os.path.expanduser("~"), ".tess-cloud-cache")
cache = diskcache.Cache(directory=CACHEDIR)


def get_boto3_client():
    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def _load_manifest():
    """Returns `tess/public/manifest.txt.gz` as a dataframe.

    This function is slow!  Use `load_manifest_lookup` for a cached lookup table.
    """
    s3c = get_boto3_client()
    obj = s3c.get_object(Bucket="stpubdata", Key="tess/public/manifest.txt.gz")
    df = pd.read_fwf(
        io.BytesIO(obj["Body"].read()),
        compression="gzip",
        names=["modified_date", "modified_time", "size", "path"],
    )
    return df


@lru_cache(maxsize=None)  # in-memory cache
def _load_ffi_manifest():
    """Returns the calibrated FFI files listed in `tess/public/manifest.txt.gz` as a dataframe."""
    df = _load_manifest()
    # Filter out the calibrated FFI FITS files
    ffi_files = df[df.path.str.endswith("ffic.fits")]
    return ffi_files


@cache.memoize(expire=86400)  # persistent on-disk cache
def _load_manifest_lookup() -> dict:
    """Returns a hashtable mapping filename => S3 URI"""
    ffi_files = _load_ffi_manifest()
    # Make a lookup hashtable which maps filename => path
    lookup = dict(zip(ffi_files.path.str.split("/").str[-1], ffi_files.path))
    return lookup


@lru_cache(maxsize=None)  # faster in-memory cache
def load_manifest_lookup() -> dict:
    return _load_manifest_lookup()


def get_s3_uri(filename: str) -> str:
    """Returns the S3 URI of a TESS data product given its filename."""
    lookup = load_manifest_lookup()
    return "s3://stpubdata/" + lookup[filename]


def list_images(sector: int, camera: int = None, ccd: int = None):
    """Returns a list of the FFIs for a given sector/camera/ccd."""
    if camera is None:
        camera = "\d"  # regex
    if ccd is None:
        ccd = "\d"  # regex
    ffi_files = _load_ffi_manifest()
    mask = ffi_files.path.str.match(
        f".*tess(\d+)-s{sector:04d}-{camera}-{ccd}-\d+-._ffic.fits"
    )
    return ffi_files[mask].path.str.split("/").str[-1].values.tolist()
