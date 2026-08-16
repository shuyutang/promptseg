"""File formats, behind one interface.

:mod:`media.base` defines what a loaded file looks like to the rest of the app;
:mod:`media.dicom_source` and :mod:`media.raster_source` implement it for DICOM
and for ordinary pictures; :mod:`media.loader` decides which is which and puts a
folder in reading order.
"""
