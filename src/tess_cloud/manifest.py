from functools import lru_cache
import os

import diskcache
import pandas as pd

from . import TESS_S3_BUCKET

# Setup the disk cache
CACHEDIR = os.path.join(os.path.expanduser("~"), ".tess-cloud-cache")
cache = diskcache.Cache(directory=CACHEDIR)


@cache.memoize(expire=86400)  # on-disk cache
def load_manifest():
    """Returns a DataFrame listing all the TESS files available on AWS S3."""
    return pd.read_fwf('s3://stpubdata/tess/public/manifest.txt.gz',
                       compression='gzip',
                       names=['modified_date', 'modified_time', 'size', 'path'])


@lru_cache  # in-memory cache
def get_cloud_uri(filename):
    """Returns the AWS S3 URI of a TESS file."""
    manifest = load_manifest()
    result = manifest[manifest.path.str.endswith(filename)]
    if len(result) == 1:
        return "s3://" + TESS_S3_BUCKET + "/" + result.path.iloc[0]
    elif len(result) > 1:
        raise ValueError("AWS Manifest contains duplicate images")
    return None