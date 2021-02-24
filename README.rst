tess-cloud
==========

**Analyze TESS Full Frame Images using AWS S3.**

|pypi|

.. |pypi| image:: https://img.shields.io/pypi/v/tess-cloud
                :target: https://pypi.python.org/pypi/tess-cloud


`tess-cloud` is a user-friendly package which provides fast access to sections of TESS Full-Frame Image (FFI) data.
It uses the AWS S3 public data set for TESS to access those parts of an FFI that are required
to obtain data cut-outs.

Installation
------------

.. code-block:: bash

    python -m pip install tess-cloud

Example use
-----------

Obtain a Target Pixel File for a stationary object:

.. code-block:: python

    >>> from tess_cloud import cutout
    >>> cutout("Alpha Cen", shape=(10, 10))
    TargetPixelFile("Alpha Cen")


Obtain a Target Pixel File centered on a moving asteroid:

.. code-block:: python

    >>> from tess_cloud import cutout_asteroid
    >>> cutout_asteroid("Vesta", start="2019-04-28", stop="2019-06-28)
    TargetPixelFile("Vesta")


Quickly download the header of an FFI:

.. code-block:: python

    >>> from tess_cloud import cutout_header
    >>> cutout_header(url, ext=0)
    FitsHeader


Documentation
-------------

Coming soon!


Similar services
----------------

`TESScut <https://mast.stsci.edu/tesscut/>`_ is an excellent API service which allows cut outs
to be obtained for stationary objects.  Tess-cloud provides an alternative implementation of this
service by leveraging the TESS public data set on AWS S3.

Compared to TESScut, the goal of tess-cloud is provide an alternative way to obtain cut-outs which
does not require a central API service, but can instead be run on a local machine or in the cloud.
At this time tess-cloud is an experiment, we recommend that you keep using TESScut for now!
