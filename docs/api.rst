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
   m2
   max
   mean
   min
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
.. autoclass:: count
.. autoclass:: any
.. autoclass:: sum
.. autoclass:: m2
.. autoclass:: min
.. autoclass:: max
.. autoclass:: mean
.. autoclass:: var
.. autoclass:: std
.. autoclass:: count_cat
.. autoclass:: summary

.. automodule:: datashader.transfer_functions
   :members:
