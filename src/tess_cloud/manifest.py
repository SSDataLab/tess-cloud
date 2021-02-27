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


@cache.memoize(expire=86400)  # on-disk cache
def load_manifest_lookup():
    """Returns a hashtable mapping filename => S3 URI"""
    s3c = get_boto3_client()
    obj = s3c.get_object(Bucket="stpubdata", Key="tess/public/manifest.txt.gz")
    df = pd.read_fwf(io.BytesIO(obj['Body'].read()),
                     compression='gzip',
                     names=['modified_date', 'modified_time', 'size', 'path'])
    # Filter out the FITS files
    fits_files = df[df.path.str.contains("fits")]
    # Make a lookup hashtable which maps filename => path
    lookup = dict(zip(fits_files.path.str.split("/").str[-1], fits_files.path))
    return lookup


@lru_cache  # in-memory cache
def get_cloud_uri(filename):
    """Returns the S3 URI of a TESS data product given its filename."""
    lookup = load_manifest_lookup()
    return "s3://stpubdata/" + lookup[filename]
