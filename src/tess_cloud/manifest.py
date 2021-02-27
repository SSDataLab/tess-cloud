# TODO: Feb 27: MANIFEST on S3 doesn't appear to have been updated since sector 26.  Caution!
from functools import lru_cache
import io
import os

import boto3
from botocore import UNSIGNED
from botocore.config import Config

import diskcache
import pandas as pd

from . import TESS_S3_BUCKET

# Setup the disk cache
CACHEDIR = os.path.join(os.path.expanduser("~"), ".tess-cloud-cache")
cache = diskcache.Cache(directory=CACHEDIR)


def get_boto3_client():
    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


@lru_cache(maxsize=None)  # in-memory cache
def _load_manifest_table():
    """Returns `tess/public/manifest.txt.gz` as a dataframe.
    
    This function is slow!  Use `load_manifest_lookup` for a cached lookup table.
    """
    s3c = get_boto3_client()
    obj = s3c.get_object(Bucket="stpubdata", Key="tess/public/manifest.txt.gz")
    df = pd.read_fwf(io.BytesIO(obj['Body'].read()),
                     compression='gzip',
                     names=['modified_date', 'modified_time', 'size', 'path'])
    return df    


@cache.memoize(expire=86400)  #persistent on-disk cache
def _load_manifest_lookup() -> dict:
    """Returns a hashtable mapping filename => S3 URI"""
    df = _load_manifest_table()
    # Filter out the calibrared FFI FITS files
    fits_files = df[df.path.str.endswith("ffic.fits")]
    # Make a lookup hashtable which maps filename => path
    lookup = dict(zip(fits_files.path.str.split("/").str[-1], fits_files.path))
    return lookup


@lru_cache(maxsize=None)  # faster in-memory cache
def load_manifest_lookup() -> dict:
    return _load_manifest_lookup()


def get_cloud_uri(filename: str) -> str:
    """Returns the S3 URI of a TESS data product given its filename."""
    lookup = load_manifest_lookup()
    return "s3://stpubdata/" + lookup[filename]
