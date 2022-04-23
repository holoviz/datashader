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

.. autosummary::

   Point
   Triangles
   PolygonGeom
   QuadMeshRaster
   QuadMeshRectilinear
   QuadMeshCurvilinear
   LineAxis0
   LineAxis0Multi
   LinesAxis1
   LinesAxis1XConstant
   LinesAxis1YConstant
   LinesAxis1Ragged
   LineAxis1Geometry
   AreaToZeroAxis0
   AreaToZeroAxis0Multi
   AreaToZeroAxis1
   AreaToZeroAxis1XConstant
   AreaToZeroAxis1YConstant
   AreaToZeroAxis1Ragged
   AreaToLineAxis0
   AreaToLineAxis0Multi
   AreaToLineAxis1
   AreaToLineAxis1XConstant
   AreaToLineAxis1YConstant
   AreaToLineAxis1Ragged

Reductions
----------

.. currentmodule:: datashader.reductions
.. autosummary::

   any
   count
   by
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

**Categorizers**

   category_binning
   category_modulo

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
