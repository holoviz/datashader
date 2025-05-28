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
   Canvas.area
   Canvas.polygons
   Canvas.quadmesh

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
   where

The table below indicates which ``Reduction`` classes are supported on the CPU (e.g. using
``pandas``), on CPU with Dask (e.g. using ``dask.dataframe``), on the GPU (e.g. using ``cudf``),
and on the GPU with Dask (e.g. using ``dask-cudf``). The final two columns indicate which reductions
support antialiased lines and which can be used as the ``selector`` in a
:class:`~datashader.reductions.where` reduction.

.. csv-table::
   :file: reduction.csv
   :header-rows: 1

The :class:`~datashader.reductions.mode` reduction is not listed in the table and can only be used
with ``Canvas.raster``. A :class:`~datashader.reductions.by` reduction supports anything that its
contained reduction (that is applied separately to each category) supports.

**Categorizers**

.. autosummary::

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
.. autoclass:: Triangles
.. autoclass:: PolygonGeom
.. autoclass:: QuadMeshRaster
.. autoclass:: QuadMeshRectilinear
.. autoclass:: QuadMeshCurvilinear
.. autoclass:: LineAxis0
.. autoclass:: LineAxis0Multi
.. autoclass:: LinesAxis1
.. autoclass:: LinesAxis1XConstant
.. autoclass:: LinesAxis1YConstant
.. autoclass:: LinesAxis1Ragged
.. autoclass:: LineAxis1Geometry
.. autoclass:: AreaToZeroAxis0
.. autoclass:: AreaToZeroAxis0Multi
.. autoclass:: AreaToZeroAxis1
.. autoclass:: AreaToZeroAxis1XConstant
.. autoclass:: AreaToZeroAxis1YConstant
.. autoclass:: AreaToZeroAxis1Ragged
.. autoclass:: AreaToLineAxis0
.. autoclass:: AreaToLineAxis0Multi
.. autoclass:: AreaToLineAxis1
.. autoclass:: AreaToLineAxis1XConstant
.. autoclass:: AreaToLineAxis1YConstant
.. autoclass:: AreaToLineAxis1Ragged

.. currentmodule:: datashader.reductions
.. autoclass:: any
.. autoclass:: by
.. autoclass:: count
.. autoclass:: count_cat
.. autoclass:: first
.. autoclass:: first_n
.. autoclass:: last
.. autoclass:: last_n
.. autoclass:: m2
.. autoclass:: max
.. autoclass:: max_n
.. autoclass:: mean
.. autoclass:: min
.. autoclass:: min_n
.. autoclass:: mode
.. autoclass:: std
.. autoclass:: sum
.. autoclass:: summary
.. autoclass:: var
.. autoclass:: where

.. automodule:: datashader.transfer_functions
   :members:
