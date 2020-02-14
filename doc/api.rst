API
===

Entry Points
------------

.. currentmodule:: datashader

**Canvas**

.. autosummary::

   Canvas
   Canvas.line
   Canvas.points
   Canvas.raster
   Canvas.trimesh
   Canvas.validate

.. currentmodule:: datashader

**Pipeline**

.. autosummary::

   Pipeline

Edge Bundling
-------------

.. currentmodule:: datashader.bundling
.. autosummary::

   directly_connect_edges
   hammer_bundle

Glyphs
------

.. currentmodule:: datashader.glyphs

**Point**

.. autosummary::

   Point
   Point.inputs
   Point.validate

.. currentmodule:: datashader.glyphs

**Line**

.. autosummary::

   Line
   Line.inputs
   Line.validate

Reductions
----------

.. currentmodule:: datashader.reductions
.. autosummary::

   any
   count
   count_cat
   sum_cat
   first
   last
   m2
   max
   mean
   min
   mode
   std
   sum
   summary
   var

Transfer Functions
------------------

.. currentmodule:: datashader.transfer_functions

**Image**

.. autosummary::

   Image
   Image.to_bytesio
   Image.to_pil
   
.. currentmodule:: datashader.transfer_functions

**Images**

.. autosummary::

   Images
   Images.cols

.. currentmodule:: datashader.transfer_functions

**Other**

.. autosummary::

   dynspread
   set_background
   shade
   spread
   stack

Definitions
-----------

.. currentmodule:: datashader
.. autoclass:: Canvas
.. autoclass:: Pipeline

.. currentmodule:: datashader.bundling
.. autoclass:: directly_connect_edges
.. autoclass:: hammer_bundle

.. currentmodule:: datashader.glyphs
.. autoclass:: Point
.. autoclass:: Line

.. currentmodule:: datashader.reductions
.. autoclass:: any
.. autoclass:: count
.. autoclass:: count_cat
.. autoclass:: first
.. autoclass:: last
.. autoclass:: m2
.. autoclass:: max
.. autoclass:: mean
.. autoclass:: min
.. autoclass:: mode
.. autoclass:: std
.. autoclass:: sum
.. autoclass:: summary
.. autoclass:: var

.. automodule:: datashader.transfer_functions
   :members:
