tess-cloud
==========

**Analyze the TESS open dataset in AWS S3.**

|pypi|

.. |pypi| image:: https://img.shields.io/pypi/v/tess-cloud
                :target: https://pypi.python.org/pypi/tess-cloud


`tess-cloud` is a user-friendly package which provides fast access to TESS Full-Frame Image (FFI) data in the cloud.
It builds upon `aioboto3 <https://pypi.org/project/aioboto3/>`_,
`asyncio <https://docs.python.org/3/library/asyncio.html>`_,
and `diskcache <https://pypi.org/project/diskcache/>`_
to access the `TESS data set in AWS S3 <https://registry.opendata.aws/tess/>`_
in a fast, asynchronous, and cached way.


Installation
------------

.. code-block:: bash

    python -m pip install tess-cloud


Example use
-----------

Retrieve the AWS S3 location of a TESS image:

.. code-block:: python

    >>> import tess_cloud
    >>> tess_cloud.get_s3_uri("tess2019199202929-s0014-2-3-0150-s_ffic.fits")
    "s3://stpubdata/tess/public/ffi/s0014/2019/199/2-3/tess2019199202929-s0014-2-3-0150-s_ffic.fits"


List the images of a TESS sector:

.. code-block:: python

    >>> tess_cloud.list_images(sector=5, camera=2, ccd=3)
    <TessImageList>


Read a TESS image from S3 into local memory:

.. code-block:: python

    >>> from tess_cloud import TessImage
    >>> img = TessImage("tess2019199202929-s0014-2-3-0150-s_ffic.fits")
    >>> img.read()
    <astropy.io.fits.HDUList>


Read only the header of a TESS image into local memory:

.. code-block:: python

    >>> img.read_header(ext=1)
    <astropy.io.fits.FitsHeader>


Cutout a Target Pixel File for a stationary object:

.. code-block:: python

    >>> from tess_cloud import cutout
    >>> cutout("Alpha Cen", shape=(10, 10))
    TargetPixelFile("Alpha Cen")


Cutout a Target Pixel File centered on a moving asteroid:

.. code-block:: python

    >>> from tess_cloud import cutout_asteroid
    >>> cutout_asteroid("Vesta", start="2019-04-28", stop="2019-06-28)
    TargetPixelFile("Vesta")


Documentation
-------------

Coming soon!


Similar services
----------------

`TESScut <https://mast.stsci.edu/tesscut/>`_ is an excellent API service which allows cut outs
to be obtained for stationary objects.  Tess-cloud provides an alternative implementation of this
service by leveraging the TESS public data set on AWS S3.

At this time tess-cloud is an experiment, we recommend that you keep using TESScut for now!
