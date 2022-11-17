Version 0.14.3 (2022-11-17)
---------------------------

This release fixes a bug related to spatial indexing of ``spatialpandas.GeoDataFrames``, and introduces enhancements to antialiased lines, benchmarking and GPU support.

Thanks to first-time contributors `@eriknw <https://github.com/eriknw>`_ and `@raybellwaves <https://github.com/raybellwaves>`_, and also `@ianthomas23 <https://github.com/ianthomas23>`_ and `@maximlt <https://github.com/maximlt>`_.

Enhancements:

* Improvements to antialiased lines:

  - Fit antialiased line code within usual numba/dask framework (`#1142 <https://github.com/holoviz/datashader/pull/1142>`_)
  - Refactor stage 2 aggregation for antialiased lines (`#1145 <https://github.com/holoviz/datashader/pull/1145>`_)
  - Support compound reductions for antialiased lines on the CPU (`#1146 <https://github.com/holoviz/datashader/pull/1146>`_)

* New benchmark framework:

  - Add benchmarking framework using ``asv`` (`#1120 <https://github.com/holoviz/datashader/pull/1120>`_)
  - Add ``cudf``, ``dask`` and ``dask-cudf`` ``Canvas.line`` benchmarks (`#1140 <https://github.com/holoviz/datashader/pull/1140>`_)

* Improvements to GPU support:

  - Cupy implementation of eq_hist (`#1129 <https://github.com/holoviz/datashader/pull/1129>`_)

* Improvements to documentation:

  - Fix markdown syntax for link (`#1119 <https://github.com/holoviz/datashader/pull/1119>`_)
  - DOC: add text link to https://examples.pyviz.org/datashader_dashboard (`#1123 <https://github.com/holoviz/datashader/pull/1123>`_)

* Improvements to dependency management (`#1111 <https://github.com/holoviz/datashader/pull/1111>`_, `#1116 <https://github.com/holoviz/datashader/pull/1116>`_)

* Improvements to CI (`#1132 <https://github.com/holoviz/datashader/pull/1132>`_, `#1135 <https://github.com/holoviz/datashader/pull/1135>`_, `#1136 <https://github.com/holoviz/datashader/pull/1136>`_, `#1137 <https://github.com/holoviz/datashader/pull/1137>`_, `#1143 <https://github.com/holoviz/datashader/pull/1143>`_)

Bug fixes:

*  Ensure spatial index ``_sindex`` is retained on dataframe copy (`#1122 <https://github.com/holoviz/datashader/pull/1122>`_)

Version 0.14.2 (2022-08-10)
---------------------------

This is a bug fix release to fix an important divide by zero bug in antialiased lines, along with improvements to documentation and handling of dependencies.

Thanks to `@ianthomas23 <https://github.com/ianthomas23>`_ and `@adamjhawley <https://github.com/adamjhawley>`_.

Enhancements:

* Improvements to documentation:

  - Fix links in docs when viewed in browser (`#1102 <https://github.com/holoviz/datashader/pull/1102>`_)
  - Add release notes (`#1108 <https://github.com/holoviz/datashader/pull/1108>`_)

* Improvements to handling of dependencies:

  - Correct dask and bokeh dependencies (`#1104 <https://github.com/holoviz/datashader/pull/1104>`_)
  - Add requests as an install dependency (`#1105 <https://github.com/holoviz/datashader/pull/1105>`_)
  - Better handle returned dask npartitions in tests (`#1107 <https://github.com/holoviz/datashader/pull/1107>`_)

Bug fixes:

* Fix antialiased line divide by zero bug (`#1099 <https://github.com/holoviz/datashader/pull/1099>`_)

Version 0.14.1 (2022-06-21)
---------------------------

This release provides a number of important bug fixes and small enhancements from Ian Thomas along with infrastructure improvements from Maxime Liquet and new reductions from `@tselea <https://github.com/tselea>`_.

Enhancements:

* Improvements to `antialiased lines <https://datashader.org/user_guide/Timeseries.html#antialiasing>`_:

  - Support antialiased lines for categorical aggregates (`#1081 <https://github.com/holoviz/datashader/pull/1081>`_, `#1083 <https://github.com/holoviz/datashader/pull/1083>`_)
  - Correctly handle NaNs in antialiased line coordinates (`#1097 <https://github.com/holoviz/datashader/pull/1097>`_)

* Improvements to ``rescale_discrete_levels`` for ``how='eq_hist'``:

  - Correct implementation of ``rescale_discrete_levels`` (`#1078 <https://github.com/holoviz/datashader/pull/1078>`_)
  - Check before calling ``rescale_discrete_levels`` (`#1085 <https://github.com/holoviz/datashader/pull/1085>`_)
  - Remove empty histogram bins in ``eq_hist`` (`#1094 <https://github.com/holoviz/datashader/pull/1094>`_)

* Implementation of first and last reduction (`#1093 <https://github.com/holoviz/datashader/pull/1093>`_) for data types other than raster.

Bug fixes:

* Do not snap trimesh vertices to pixel grid (`#1092 <https://github.com/holoviz/datashader/pull/1092>`_)
* Correctly orient (y, x) arrays for xarray (`#1095 <https://github.com/holoviz/datashader/pull/1095>`_)
* Infrastructure/build fixes (`#1080 <https://github.com/holoviz/datashader/pull/1080>`_, `#1089 <https://github.com/holoviz/datashader/pull/1089>`_, `#1096 <https://github.com/holoviz/datashader/pull/1096>`_)

Version 0.14.0 (2022-04-25)
---------------------------

This release has been nearly a year in the making, with major new contributions from Ian Thomas, Thuy Do Thi Minh, Simon Høxbro Hansen, Maxime Liquet, and James Bednar, and additional support from Andrii Oriekhov, Philipp Rudiger, and Ajay Thorve.

Enhancements:

- Full support for `antialiased lines <https://datashader.org/user_guide/Timeseries.html#antialiasing>`_ of specified width (`#1048 <https://github.com/holoviz/datashader/pull/1048>`_, `#1072 <https://github.com/holoviz/datashader/pull/1072>`_). Previous antialiasing support was limited to single-pixel lines and certain floating-point reduction functions. Now supports arbitrary widths and arbitrary reduction functions, making antialiasing fully supported. Performance ranges from 1.3x to 14x slower than the simplest zero-width implementation; see `benchmarks <https://github.com/holoviz/datashader/pull/1072>`_.
- Fixed an issue with visibility on zoomed-in points plots and on overlapping line plots that was first reported in 2017, with a new option ``rescale_discrete_levels`` for ``how='eq_hist'`` (`#1055 <https://github.com/holoviz/datashader/pull/1055>`_)
- Added a `categorical color_key for 2D <https://datashader.org/getting_started/Pipeline.html#colormapping-2d-categorical-data>`_ (unstacked) aggregates (`#1020 <https://github.com/holoviz/datashader/pull/1020>`_), for producing plots where each pixel has at most one category value
- Improved docs:

  * A brand new `polygons guide <https://datashader.org/user_guide/Polygons.html>`_ (`#1071 <https://github.com/holoviz/datashader/pull/1071>`_)
  * A new guide to `3D aggregations <https://datashader.org/getting_started/Pipeline.html#id1>`_ using ``by`` now  documenting using ``categorizer`` objects to do 3D numerical binning (`#1071 <https://github.com/holoviz/datashader/pull/1071>`_)
  * Moved documentation for `spreading <https://datashader.org/getting_started/Pipeline.html#spreading>`_ to its own section so it can be presented at the right pipeline stage (was mixed up with colormapping before) (`#1071 <https://github.com/holoviz/datashader/pull/1071>`_)
  * Added `rescale_discrete_levels example <https://datashader.org/getting_started/Pipeline.html#transforming-data-values-for-colormapping>`_ (`#1071 <https://github.com/holoviz/datashader/pull/1071>`_)
  * Other misc doc cleanup (`#1035 <https://github.com/holoviz/datashader/pull/1035>`_, `#1037 <https://github.com/holoviz/datashader/pull/1037>`_, `#1058 <https://github.com/holoviz/datashader/pull/1058>`_, `#1074 <https://github.com/holoviz/datashader/pull/1074>`_, `#1077 <https://github.com/holoviz/datashader/pull/1077>`_)

Bugfixes:

- Fixed details of the raster coordinate calculations to match other primitives, making it simpler to overlay separately rendered results (`#959 <https://github.com/holoviz/datashader/pull/959>`_, `#1046 <https://github.com/holoviz/datashader/pull/1046>`_)
- Various fixes and extensions for cupy/CUDA, e.g. to use cuda for category_binning, spread, and dynspread, including cupy.interp where appropriate (`#1015 <https://github.com/holoviz/datashader/pull/1015>`_, `#1016 <https://github.com/holoviz/datashader/pull/1016>`_, `#1044 <https://github.com/holoviz/datashader/pull/1044>`_, `#1050 <https://github.com/holoviz/datashader/pull/1050>`_, `#1060 <https://github.com/holoviz/datashader/pull/1060>`_)
- Infrastructure/build/ecosystem fixes (`#1022 <https://github.com/holoviz/datashader/pull/1022>`_, `#1025 <https://github.com/holoviz/datashader/pull/1025>`_, `#1027 <https://github.com/holoviz/datashader/pull/1027>`_, `#1036 <https://github.com/holoviz/datashader/pull/1036>`_, `#1045 <https://github.com/holoviz/datashader/pull/1045>`_, `#1049 <https://github.com/holoviz/datashader/pull/1049>`_, `#1050 <https://github.com/holoviz/datashader/pull/1050>`_, `#1057 <https://github.com/holoviz/datashader/pull/1057>`_, `#1061 <https://github.com/holoviz/datashader/pull/1061>`_, `#1062 <https://github.com/holoviz/datashader/pull/1062>`_, `#1063 <https://github.com/holoviz/datashader/pull/1063>`_, `#1064 <https://github.com/holoviz/datashader/pull/1064>`_)

Compatibility:

- ``Canvas.line()`` option ``antialias=True`` is now deprecated; use ``line_width=1`` (or another nonzero value) instead. (`#1048 <https://github.com/holoviz/datashader/pull/1048>`_)
- Removed long-deprecated ``bokeh_ext.py`` (`#1059 <https://github.com/holoviz/datashader/pull/1059>`_)
- Dropped support for Python 2.7 (actually already dropped from the tests in Datashader 0.12) and 3.6 (no longer supported by many downstream libraries like rioxarray, but several of them are not properly declaring that restriction, making 3.6 much more difficult to support.) (`#1033 <https://github.com/holoviz/datashader/pull/1033>`_)
- Now tested on Python 3.7, 3.8, 3.9, and 3.10. (`#1033 <https://github.com/holoviz/datashader/pull/1033>`_)

Version 0.13.0 (2021-06-10)
---------------------------

Thanks to Jim Bednar, Nezar Abdennur, Philipp Rudiger, and Jean-Luc Stevens.

Enhancements:

- Defined new ``dynspread metric`` based on counting the fraction of non-empty pixels that have non-empty pixels within a given radius. The resulting ``dynspread`` behavior is much more intuitive than the old behavior, which counted already-spread pixels as if they were neighbors (`#1001 <https://github.com/holoviz/datashader/pull/1001>`_)
- Added ``ds.count()`` as the default reduction for ``ds.by`` (`#1004 <https://github.com/holoviz/datashader/pull/1004>`_)

Bugfixes:

- Fixed array-bounds reading error in ``dynspread`` (`#1001 <https://github.com/holoviz/datashader/pull/1001>`_)
- Fix ``color_key`` argument for ``dsshow`` (`#986 <https://github.com/holoviz/datashader/pull/986>`_)
- Added Matplotlib output to the 3_Interactivity getting started page. (`#1009 <https://github.com/holoviz/datashader/pull/1009>`_)
- Misc docs fixes (`#1007 <https://github.com/holoviz/datashader/pull/1007>`_)
- Fix nan assignment to integer array in RaggedArray (`#1008 <https://github.com/holoviz/datashader/pull/1008>`_)

Compatibility:

- Any usage of ``dynspread`` with datatypes other than points should be replaced with ``spread()``, which will do what was probably intended by the original ``dynspread`` call, i.e. to make isolated lines and shapes visible. Strictly speaking, dynspread could still be useful for other glyph types if that glyph is contained entirely in a pixel, e.g. if a polygon or line segment is located within the pixel bounds, but that seems unlikely.
- Dynspread may need to have the threshold or max_px arguments updated to achieve the same spreading as in previous releases, though the new behavior is normally going to be more useful than the old.

Version 0.12.1 (2021-03-22)
---------------------------

Major release with new features that should really be considered part of the upcoming 0.13 release; please treat all the new features as experimental in this release due to it being officially a minor release (unintentionally).

Massive thanks to these contributors for substantial new functionality:

- Nezar Abdennur (nvictus), Trevor Manz, and Thomas Caswell for their contributions to the new ``dsshow()`` support for using Datashader as a Matplotlib Artist, providing seamless interactive Matplotlib+Datashader plots.
- Oleg Smirnov for ``category_modulo`` and ``category_binning`` for ``by()``, making categorical plots vastly more powerful.
- Jean-Luc Stevens for ``spread`` and ``dynspread`` support for numerical aggregate arrays and not just RGB images, allowing isolated datapoints to be made visible while still supporting hover, colorbars, and other plot features that depend on the numeric aggregate values.
- Valentin Haenel for the initial anti-aliased line drawing support (still experimental).

Thanks to Jim Bednar, Philipp Rudiger, Peter Roelants, Thuy Do Thi Minh, Chris Ball, and Jean-Luc Stevens for maintenance and other contributions.

New features:

- Expanded (and transposed) performance guide table (`#961 <https://github.com/holoviz/datashader/pull/961>`_)
- Add ``category_modulo`` and ``category_binning`` for grouping numerical values into categories using by() (`#927 <https://github.com/holoviz/datashader/pull/927>`_)
- Support spreading for numerical (non-RGB) aggregate arrays (`#771 <https://github.com/holoviz/datashader/pull/771>`_, `#954 <https://github.com/holoviz/datashader/pull/954>`_)
- Xiaolin Wu anti-aliased line drawing, enabled by adding ``antialias=True`` to the ``Canvas.line()`` method call. Experimental; currently restricted to ``sum`` and ``max`` reductions ant only supporting a single-pixel line width. (`#916 <https://github.com/holoviz/datashader/pull/916>`_)
- Improve Dask performance issue using a tree reduction (`#926 <https://github.com/holoviz/datashader/pull/926>`_)

Bugfixes:

- Fix for xarray 0.17 raster files, supporting various nodata conventions (`#991 <https://github.com/holoviz/datashader/pull/991>`_)
- Fix RaggedArray tests to keep up with Pandas test suite changes (`#982 <https://github.com/holoviz/datashader/pull/982>`_, `#993 <https://github.com/holoviz/datashader/pull/993>`_)
- Fix out-of-bounds error on Points aggregation (`#981 <https://github.com/holoviz/datashader/pull/981>`_)
- Fix CUDA issues (`#973 <https://github.com/holoviz/datashader/pull/973>`_)
- Fix Xarray handling (`#971 <https://github.com/holoviz/datashader/pull/971>`_)
- Disable the interactivity warning on the homepage (`#983 <https://github.com/holoviz/datashader/pull/983>`_)

Compatibility:

- Drop deprecated modules ``ds.geo`` (moved to ``xarray_image``) and ``ds.spatial`` (moved to ``SpatialPandas``) (`#955 <https://github.com/holoviz/datashader/pull/955>`_)

Version 0.12.0 (2021-01-07)
---------------------------

No release notes produced.

Version 0.11.1 (2020-08-16)
---------------------------

This release is primarily a compatibility release for newer versions of Rapids cuDF and Numba versions along with a small number of bug fixes. With contributions from `@jonmmease <https://github.com/jonmmease>`_, `@stuartarchibald <https://github.com/stuartarchibald>`_, `@AjayThorve <https://github.com/AjayThorve>`_, `@kebowen730 <https://github.com/kebowen730>`_, `@jbednar <https://github.com/jbednar>`_ and `@philippjfr <https://github.com/philippjfr>`_.

- Fixes support for cuDF 0.13 and Numba 0.48 (`#933 <https://github.com/holoviz/datashader/pull/933>`_)
- Fixes for cuDF support on Numba>=0.51 (`#934 <https://github.com/holoviz/datashader/pull/934>`_, `#947 <https://github.com/holoviz/datashader/pull/947>`_)
- Fixes tile generation using aggregators with output of boolean dtype (`#949 <https://github.com/holoviz/datashader/pull/949>`_)
- Fixes for CI and build infrastructure (`#935 <https://github.com/holoviz/datashader/pull/935>`_, `#948 <https://github.com/holoviz/datashader/pull/948>`_, `#951 <https://github.com/holoviz/datashader/pull/951>`_)
- Updates to docstrings (b1349e3, `#950 <https://github.com/holoviz/datashader/pull/950>`_)

Version 0.11.0 (2020-05-25)
---------------------------

This release includes major contributions from `@maihde <https://github.com/maihde>`_ (generalizing ``count_cat`` to ``by`` span for colorize), `@jonmmease <https://github.com/jonmmease>`_ (Dask quadmesh support), `@philippjfr <https://github.com/philippjfr>`_ and `@jbednar <https://github.com/jbednar>`_ (count_cat/by/colorize/docs/bugfixes), and Barry Bragg, Jr. (TMS tileset speedups).

New features (see ``getting_started/2_Pipeline.ipynb`` for examples):

- New ``by()`` categorical aggregator, extending ``count_cat`` to work with other reduction functions, no longer just ``count``. Allows binning of aggregates separately per category value, so that you can compare how that aggregate is affected by category value. (`#875 <https://github.com/holoviz/datashader/pull/875>`_, `#902 <https://github.com/holoviz/datashader/pull/902>`_, `#904 <https://github.com/holoviz/datashader/pull/904>`_, `#906 <https://github.com/holoviz/datashader/pull/906>`_). See example in the `holoviews docs <http://dev.holoviews.org/user_guide/Large_Data.html#Multidimensional-plots>`_.
- Support for negative and zero values in ``tf.shade`` for categorical aggregates. (`#896 <https://github.com/holoviz/datashader/pull/896>`_, `#909 <https://github.com/holoviz/datashader/pull/909>`_, `#910 <https://github.com/holoviz/datashader/pull/910>`_, `#908 <https://github.com/holoviz/datashader/pull/908>`_)
- Support for ``span`` in _colorize(). (`#875 <https://github.com/holoviz/datashader/pull/875>`_, `#910 <https://github.com/holoviz/datashader/pull/910>`_)
- Support for Dask-based quadmesh rendering for rectilinear and curvilinear mesh types (`#885 <https://github.com/holoviz/datashader/pull/885>`_, `#913 <https://github.com/holoviz/datashader/pull/913>`_)
- Support for GPU-based raster mesh rendering via ``Canvas.quadmesh`` (`#872 <https://github.com/holoviz/datashader/pull/872>`_)
- Faster TMS tileset generation (`#886 <https://github.com/holoviz/datashader/pull/886>`_)
- Expanded performance guide (`#868 <https://github.com/holoviz/datashader/pull/868>`_)

Bugfixes:

- Misc bugfixes and improvements (`#874 <https://github.com/holoviz/datashader/pull/874>`_, `#882 <https://github.com/holoviz/datashader/pull/882>`_, `#888 <https://github.com/holoviz/datashader/pull/888>`_, `#889 <https://github.com/holoviz/datashader/pull/889>`_, `#890 <https://github.com/holoviz/datashader/pull/890>`_, `#891 <https://github.com/holoviz/datashader/pull/891>`_)

Compatibility (breaking changes and deprecations):

- To allow negative-valued aggregates, count_cat now weights categories according to how far they are from the minimum aggregate value observed, while previously they were referenced to zero. Previous behavior can be restored by passing ``color_baseline=0`` to ``count_cat`` or ``by``
- ``count_cat`` is now deprecated and removed from the docs; use ``by(..., count())`` instead.
- Result of a ``count()`` aggregation is now ``uint32`` not ``int32`` to distinguish counts from other aggregation types (`#910 <https://github.com/holoviz/datashader/pull/910>`_).
- tf.shade now only treats zero values as missing for ``count`` aggregates (``uint``; zero is otherwise a valid value distinct from NaN (`#910 <https://github.com/holoviz/datashader/pull/910>`_).
- ``alpha`` is now respected as the upper end of the alpha range for both _colorize() and _interpolate() in tf.shade; previously only _interpolate respected it.
- Added new nansum_missing utility for working with Numpy>1.9, where nansum no longer returns NaN for all-NaN values.
- ds.geo and ds.spatial modules are now deprecated; their contents have moved to xarray_spatial and spatialpandas, respectively.  (`#894 <https://github.com/holoviz/datashader/pull/894>`_)

Download and install: https://datashader.org/getting_started

Version 0.10.0 (2020-01-21)
---------------------------

This release includes major contributions from `@jonmmease <https://github.com/jonmmease>`_ (polygon rendering, spatialpandas), along with contributions from `@philippjfr <https://github.com/philippjfr>`_ and `@brendancol <https://github.com/brendancol>`_ (bugfixes), and `@jbednar <https://github.com/jbednar>`_ (docs, warnings, and import times).

New features:

- Polygon (and points and lines) rendering for spatialpandas extension arrays (`#826 <https://github.com/holoviz/datashader/pull/826>`_, `#853 <https://github.com/holoviz/datashader/pull/853>`_)
- Quadmesh GPU support (`#861 <https://github.com/holoviz/datashader/pull/861>`_)
- Much faster import times (`#863 <https://github.com/holoviz/datashader/pull/863>`_)
- New table in docs listing glyphs supported for each data library (`#864 <https://github.com/holoviz/datashader/pull/864>`_, `#867 <https://github.com/holoviz/datashader/pull/867>`_)
- Support for remote Parquet filesystems (`#818 <https://github.com/holoviz/datashader/pull/818>`_, `#866 <https://github.com/holoviz/datashader/pull/866>`_)

Bugfixes and compatibility:

- Misc bugfixes and improvements (`#844 <https://github.com/holoviz/datashader/pull/844>`_, `#860 <https://github.com/holoviz/datashader/pull/860>`_, `#866 <https://github.com/holoviz/datashader/pull/866>`_)
- Fix warnings and deprecations in tests (`#859 <https://github.com/holoviz/datashader/pull/859>`_)
- Fix Canvas.raster (padding, mode buffers, etc. `#862 <https://github.com/holoviz/datashader/pull/862>`_)

Download and install: https://datashader.org/getting_started

Version 0.9.0 (2019-12-08)
--------------------------

This release includes major contributions from `@jonmmease <https://github.com/jonmmease>`_ (GPU support), along with contributions from `@brendancol <https://github.com/brendancol>`_ (viewshed speedups), `@jbednar <https://github.com/jbednar>`_ (docs), and `@jsignell <https://github.com/jsignell>`_ (examples, maintenance, website).

New features:

- Support for CUDA GPU dataframes (cudf and dask_cudf) (`#794 <https://github.com/holoviz/datashader/pull/794>`_, `#793 <https://github.com/holoviz/datashader/pull/793>`_, `#821 <https://github.com/holoviz/datashader/pull/821>`_, `#841 <https://github.com/holoviz/datashader/pull/841>`_, `#842 <https://github.com/holoviz/datashader/pull/842>`_)
- Documented new quadmesh support (renaming user guide section 5_Rasters to 5_Grids to reflect the more-general grid support) (`#805 <https://github.com/holoviz/datashader/pull/805>`_)

Bugfixes and compatibility:

- Avoid double-counting line segments that fit entirely into a single rendered pixel (`#839 <https://github.com/holoviz/datashader/pull/839>`_)
- Improved geospatial toolbox, including 75X speedups to viewshed algorithm (`#811 <https://github.com/holoviz/datashader/pull/811>`_, `#824 <https://github.com/holoviz/datashader/pull/824>`_, `#844 <https://github.com/holoviz/datashader/pull/844>`_)

Version 0.8.0 (2019-10-08)
--------------------------

This release includes major contributions from `@jonmmease <https://github.com/jonmmease>`_ (quadmesh and filled-area support), `@brendancol <https://github.com/brendancol>`_ (geospatial toolbox, tile previewer), `@philippjfr <https://github.com/philippjfr>`_ (distributed regridding, dask performance), and `@jsignell <https://github.com/jsignell>`_ (examples, maintenance, website).

New features:

- Native quadmesh (``canvas.quadmesh()`` support (for rectilinear and curvilinear grids -- 3X faster than approximating with a trimesh; `#779 <https://github.com/holoviz/datashader/pull/779>`_)
- `Filled area <https://datashader.org/user_guide/Timeseries.html#Area-plots>`_ (``canvas.area()`` support (`#734 <https://github.com/holoviz/datashader/pull/734>`_)
- Expanded `geospatial toolbox <https://datashader.org/user_guide/Geography.html>`_, with support for:

  * Zonal statistics (`#782 <https://github.com/holoviz/datashader/pull/782>`_)
  * Calculating viewshed (`#781 <https://github.com/holoviz/datashader/pull/781>`_)
  * Calculating proximity (Euclidean and other distance metrics, `#772 <https://github.com/holoviz/datashader/pull/772>`_)

- Distributed raster regridding with Dask (`#762 <https://github.com/holoviz/datashader/pull/762>`_)
- Improved dask performance (`#798 <https://github.com/holoviz/datashader/pull/798>`_, `#801 <https://github.com/holoviz/datashader/pull/801>`_)
- ``tile_previewer`` utility function (simple Bokeh-based plotting of local tile sources for debugging; `#761 <https://github.com/holoviz/datashader/pull/761>`_)

Bugfixes and compatibility:

- Compatibility with latest Numba, Intake, Pandas, and Xarray (`#763 <https://github.com/holoviz/datashader/pull/763>`_, `#768 <https://github.com/holoviz/datashader/pull/768>`_, `#791 <https://github.com/holoviz/datashader/pull/791>`_)
- Improved datetime support (`#803 <https://github.com/holoviz/datashader/pull/803>`_)
- Simplified docs (now built on Travis, and no longer requiring GeoViews) and examples (now on examples.pyviz.org)
- Skip rendering of empty tiles (`#760 <https://github.com/holoviz/datashader/pull/760>`_)
- Improved performance for point, area, and line glyphs (`#780 <https://github.com/holoviz/datashader/pull/780>`_)
- ``InteractiveImage`` and ``Pipeline`` are now deprecated; removed from examples (`#751 <https://github.com/holoviz/datashader/pull/751>`_)

Version 0.7.0 (2019-04-08)
--------------------------

This release includes major contributions from `@jonmmease <https://github.com/jonmmease>`_ (ragged array extension, SpatialPointsFrame, row-oriented line storage, dask trimesh support), `@jsignell <https://github.com/jsignell>`_ (maintenance, website), and `@jbednar <https://github.com/jbednar>`_ (Panel-based dashboard).

New features:

- Simplified `Panel <https://panel.pyviz.org>`_ based `dashboard <https://datashader.org/dashboard.html>`_ using new Param features; now only 48 lines with fewer new concepts (`#707 <https://github.com/holoviz/datashader/pull/707>`_)
- Added pandas ExtensionArray and Dask support for storing homogeneous ragged arrays (`#687 <https://github.com/holoviz/datashader/pull/687>`_)
- Added SpatialPointsFrame and updated census, osm-1billion, and osm examples to use it (`#702 <https://github.com/holoviz/datashader/pull/702>`_, `#706 <https://github.com/holoviz/datashader/pull/706>`_, `#708 <https://github.com/holoviz/datashader/pull/708>`_)
- Expanded 8_Geography.ipynb to document other geo-related functions
- Added Dask support for trimesh rendering, though computing the mesh initially still requires vertices and simplicies to fit into memory (`#696 <https://github.com/holoviz/datashader/pull/696>`_)
- Add zero-copy rendering of row-oriented line coordinates, using a new axis argument (`#694 <https://github.com/holoviz/datashader/pull/694>`_)

Bugfixes and compatibility:

- Added lnglat_to_meters to geo module; new code should import it from there (`#708 <https://github.com/holoviz/datashader/pull/708>`_)

Version 0.6.9 (2019-01-29)
--------------------------

This release includes major contributions from `@jonmmease <https://github.com/jonmmease>`_ (fixing several long-standing bugs), `@jlstevens <https://github.com/jlstevens>`_ (updating all example notebooks to use current syntax, `#685 <https://github.com/holoviz/datashader/pull/685>`_), `@jbednar <https://github.com/jbednar>`_, `@philippjfr <https://github.com/philippjfr>`_, and `@jsignell <https://github.com/jsignell>`_ (`Panel <https://panel/pyviz.org>`_-based dashboard), and `@brendancol <https://github.com/brendancol>`_ (geo utilities).

New features:

* Replaced outdated 536-line Bokeh `dashboard.py <https://github.com/pyviz/datashader/blob/ae72d237d574cbd7103a912fc84094ce10d55344/examples/dashboard/dashboard.py>`_ with 71-line Panel+HoloViews `dashboard <https://github.com/pyviz/datashader/blob/master/examples/dashboard.ipynb>`_ (`#676 <https://github.com/holoviz/datashader/pull/676>`_)
* Allow aggregating xarray objects (in addition to Pandas and Dask DataFrames) (`#675 <https://github.com/holoviz/datashader/pull/675>`_)
* Create WMTS tiles from Datashader data (`#636 <https://github.com/holoviz/datashader/pull/636>`_)
* Added various `geographic utility functions <http://datashader.org/user_guide/8_Geography.html>`_ (ndvi, slope, aspect, hillshade, mean, bump map, Perlin noise) (`#661 <https://github.com/holoviz/datashader/pull/661>`_)
* Made OpenSky data public (`#691 <https://github.com/holoviz/datashader/pull/691>`_)

Bugfixes and compatibility:

* Fix array bounds error on line glyph (`#683 <https://github.com/holoviz/datashader/pull/683>`_)
* Fixed the span argument to tf.shade (`#680 <https://github.com/holoviz/datashader/pull/680>`_)
* Fixed composite.add (for use in spreading) to clip colors rather than overflow (`#689 <https://github.com/holoviz/datashader/pull/689>`_)
* Fixed gerrymandering shape file (`#688 <https://github.com/holoviz/datashader/pull/688>`_)
* Updated to match Bokeh (`#656 <https://github.com/holoviz/datashader/pull/656>`_), Dask (`#681 <https://github.com/holoviz/datashader/pull/681>`_, `#667 <https://github.com/holoviz/datashader/pull/667>`_), Pandas/Numpy (`#697 <https://github.com/holoviz/datashader/pull/697>`_)

Version 0.6.8 (2018-09-11)
--------------------------

Minor, mostly bugfix, release with some speed improvements.

New features:

- Added Strange Attractors example (`#632 <https://github.com/holoviz/datashader/pull/632>`_)
- Major speedup: optimized dask datashape detection (`#634 <https://github.com/holoviz/datashader/pull/634>`_)

Bugfixes and compatibility:

- Silenced inappropriate warnings (`#631 <https://github.com/holoviz/datashader/pull/631>`_)
- Fixed various other bugs, including `#644 <https://github.com/holoviz/datashader/pull/644>`_
- Added handling for zero data and zero range (`#612 <https://github.com/holoviz/datashader/pull/612>`_, `#648 <https://github.com/holoviz/datashader/pull/648>`_)

Version 0.6.7 (2018-07-07)
--------------------------

Minor compatibility release.

* Supports dask >= 0.18.
* Updated installation and usage instructions

Version 0.6.6 (2018-05-20)
--------------------------

Minor bugfix release.

* Now available to install using pip (``pip install datashader``) or conda defaults (``conda install datashader``)
* InteractiveImage is now deprecated; please use the Datashader support in HoloViews instead.
* Updated installation and example instructions to use new ``datashader`` command.
* Made package building automatic, to allow more frequent releases
* Ensured transparent (not black) image is returned when there is no data to plot (thanks to Nick Xie)
* Simplified getting-started example (thanks to David Jones)
* Various fixes and compatibility updates to examples

Version 0.6.5 (2018-02-01)
--------------------------

Major release with extensive support for triangular meshes and changes to the raster API.

New features:

- Trimesh support: Rendering of irregular triangular meshes using ``Canvas.trimesh()`` (see `user guide <https://github.com/bokeh/datashader/blob/master/examples/user_guide/6_Trimesh.ipynb>`_)  (`#525 <https://github.com/holoviz/datashader/pull/525>`_, `#552 <https://github.com/holoviz/datashader/pull/552>`_)
- Added a new website at `datashader.org <https://datashader.org>`_, with new Getting Started pages and an extensive User Guide, with about 50% new material not previously in example notebooks. Built entirely from Jupyter notebooks, which can be run in the ``examples/`` directory.  Website is now complete except for sections on points (see the `nyc_taxi example <https://github.com/bokeh/datashader/blob/master/examples/topics/nyc_taxi.ipynb>`_ in the meantime).
- ``Canvas.raster()`` now accepts xarray Dataset types, not just DataArrays, with the specific DataArray selectable from the Dataset using the ``column=`` argument of a supplied aggregation function.
- ``tf.Images()`` now displays anything with an HTML representation, to allow laying out Pandas dataframes alongside datashader output.

Bugfixes and compatibility:

- Changed Raster API to match other glyph types:

  * Now accepts a reduction function via an ``agg=`` argument like ``Canvas.line()``,  ``Canvas.points()``, etc.  The previous ``downsample_method`` is still accepted for this release, but is now deprecated.
  * ``upsample_method`` is now ``interpolate``, accepting ``linear=True`` or ``linear=False``; the previous spelling is now deprecated.
  * The ``layer=`` argument previously accepted a 1-based integer index, which was confusing given the standard Python 0-based indexing elsewhere.  Changed to accept an xarray coordinate, which can be a 1-based index if that's what is defined on the array, but also works with arbitrary floating-point coordinates (e.g. for a depth parameter in an image stack).
  * Now auto-ranges in x and y when not given explicit ranges, instead of raising an error.

- Fixed various bugs, including one generating incorrect output in ``Canvas.raster(agg='mode')``

Version 0.6.4 (2017-12-05)
--------------------------

Minor compatibility release to track changes in external packages.

* Updated imports for bokeh 0.12.11 (fixes `#535 <https://github.com/holoviz/datashader/pull/535>`_), though there are issues in 0.12.11 itself and so 0.12.12 should be used instead (to be released shortly).
* Pinned pillow version on Windows (fixes `#534 <https://github.com/holoviz/datashader/pull/534>`_).

Version 0.6.3 (2017-12-01)
--------------------------

Apart from the new website, this is a minor release primarily to catch up with changes in external libraries.

New features:

* Reorganized examples directory as the basis for a completely new website at https://bokeh.github.io/datashader-docs (`#516 <https://github.com/holoviz/datashader/pull/516>`_).
* Added tf.Images() class to format multiple labeled Datashader images as a table in a Jupyter notebook, now used extensively in the new website.
* Added utility function ``dataframe_from_multiple_sequences(x_values, y_values)`` to convert large numbers of sequences stored as 2D numpy arrays to a NaN-separated pandas dataframe that can be displayed efficiently (see new example in tseries.ipynb) (`#512 <https://github.com/holoviz/datashader/pull/512>`_).
* Improved streaming support (`#520 <https://github.com/holoviz/datashader/pull/520>`_).

Bugfixes and compatibility:

* Added support for Dask 0.15 and 0.16 and pandas 0.21 (`#523 <https://github.com/holoviz/datashader/pull/523>`_, `#529 <https://github.com/holoviz/datashader/pull/529>`_) and declared minimum required Numba version.
* Improved and fixed issues with various example notebooks, primarily to update for changes in dependencies.
* Changes in network graph support: ignore id field by default to avoid surprising dependence on column name, rename directly_connect_edges to connect_edges for accuracy and conciseness.

Version 0.6.2 (2017-10-25)
--------------------------

Release with bugfixes, changes to match external libraries, and some new features.

Backwards compatibility:

* Minor changes to network graph API, e.g. to ignore weights by default in forcelayout2 (`#488 <https://github.com/holoviz/datashader/pull/488>`_)
* Fix upper-bound bin error for auto-ranged data (`#459 <https://github.com/holoviz/datashader/pull/459>`_). Previously, points falling on the upper bound of the plotted area were excluded from the plot, which was consistent with the behavior for individual grid cells, but which was confusing and misleading for the outer boundaries.  Points falling on the very outermost boundaries are now folded into the final grid cell, which should be the least surprising behavior.

New or updated examples (.ipynb files in examples/):

* `streaming-aggregation.ipynb <https://anaconda.org/jbednar/streaming-aggregation>`_: Illustrates combining incoming streams of data for display (also see `holoviews streaming <https://anaconda.org/philippjfr/working_with_streaming_data>`_).
* `landsat.ipynb <https://anaconda.org/jbednar/landsat>`_: simplified using HoloViews; now includes plots of full spectrum for each point via hovering.
* Updated and simplified census-hv-dask (now called census-congressional), census-hv, packet_capture_graph.

New features and improvements

* Updated Bokeh support to work with new bokeh 0.12.10 release (`#505 <https://github.com/holoviz/datashader/pull/505>`_)
* More options for network/graph plotting (configurable column names, control over weights usage; #488, `#494 <https://github.com/holoviz/datashader/pull/494>`_)
* For lines plots (time series, trajectory, networ graphs), switch line-clipping algorithm from Cohen-Sutherland to Liang-Barsky. The performance gains for random lines range from 50-75% improvement for a million lines. (`#495 <https://github.com/holoviz/datashader/pull/495>`_)
* Added ``tf.Images`` class to format a list of images as an HTML table (`#492 <https://github.com/holoviz/datashader/pull/492>`_)
* Faster resampling/regridding operations (`#486 <https://github.com/holoviz/datashader/pull/486>`_)

Known issues:

* examples/dashboard has not yet been updated to match other libraries, and is thus missing functionality like hovering and legends.
* A full website with documentation has been started but is not yet ready for deployment.

Version 0.6.1 (2017-09-13)
--------------------------

Minor bugfix release, primarily updating example notebooks to match API changes in external packages.

Backwards compatibility:

* Made edge bundling retain edge order, to allow indexing, and absolute coordinates, to allow overlaying on external data.
* Updated examples to show that xarray now requires dimension names to match before doing arithmetic or comparisons between arrays.

Known issues:

* If you use Jupyter notebook 5.0 (earlier or later versions should be ok), you will need to override a setting that prevents visualizations from appearing, e.g.: ``jupyter notebook --NotebookApp.iopub_data_rate_limit=100000000 census.ipynb &``
* The dashboard needs to be rewritten entirely to match current Bokeh and HoloViews releases, so that hover and legend support can be restored.

Version 0.6.0 (2017-08-19)
--------------------------

New release of features that may still be in progress, but are already usable:

* Added graph/network plotting support (still may be in flux) (`#385 <https://github.com/holoviz/datashader/pull/385>`_, `#390 <https://github.com/holoviz/datashader/pull/390>`_, `#398 <https://github.com/holoviz/datashader/pull/398>`_, `#408 <https://github.com/holoviz/datashader/pull/408>`_, `#415 <https://github.com/holoviz/datashader/pull/415>`_, `#418 <https://github.com/holoviz/datashader/pull/418>`_, `#436 <https://github.com/holoviz/datashader/pull/436>`_)
* Improved raster regridding based on gridtools and xarray (still may be in flux); no longer depends on rasterio and scikit-image (`#383 <https://github.com/holoviz/datashader/pull/383>`_, `#389 <https://github.com/holoviz/datashader/pull/389>`_, `#423 <https://github.com/holoviz/datashader/pull/423>`_)
* Significantly improved performance for dataframes with categorical fields

New examples  (.ipynb files in examples/):

* `osm-1billion <https://anaconda.org/jbednar/osm-1billion>`_: 1-billion-point OSM example, for in-core processing on a 16GB laptop.
* `edge_bundling <https://anaconda.org/jbednar/edge_bundling>`_: Plotting graphs using "edgehammer" bundling of edges to show structure.
* `packet_capture_graph <https://anaconda.org/jbednar/packet_capture_graph>`_: Laying out and visualizing network packets as a graph.

Backwards compatibility:

* Remove deprecated interpolate and colorize functions
* Made raster processing consistently use bin centers to match xarray conventions (requires recent fixes to xarray; only available on a custom channel for now) (`#422 <https://github.com/holoviz/datashader/pull/422>`_)
* Fixed various limitations and quirks for NaN values
* Made alpha scaling respect ``min_alpha`` consistently (`#371 <https://github.com/holoviz/datashader/pull/371>`_)

Known issues:

* If you use Jupyter notebook 5.0 (earlier or later versions should be ok), you will need to override a setting that prevents visualizations from appearing, e.g.: ``jupyter notebook --NotebookApp.iopub_data_rate_limit=100000000 census.ipynb &``
* The dashboard needs updating to match current Bokeh releases; most parts other than hover and legends, should be functional but it needs a rewrite to use currently recommended approaches.

Version 0.5.0 (2017-05-12)
--------------------------

Major release with extensive optimizations and new plotting-library support, incorporating 9 months of development from 5 main `contributors <https://github.com/bokeh/datashader/graphs/contributors>`_:

- Extensive optimizations for speed and memory usage, providing at least 5X improvements in speed (using the latest Numba versions) and 2X improvements in peak memory requirements.
- Added `HoloViews support <https://anaconda.org/jbednar/holoviews_datashader>`_ for flexible, composable, dynamic plotting, making it simple to switch between datashaded and non-datashaded versions of a Bokeh or Matplotlib plot.
- Added `examples/environment.yml <https://github.com/bokeh/datashader/blob/master/examples/environment.yml>`_ to make it easy to install dependencies needed to run the examples.
- Updated examples to use the now-recommended supported and fast Apache Parquet file format
- Added support for variable alpha for non-categorical aggregates, by specifying a single color rather than a list or colormap #345
- Added `datashader.utils.lnglat_to_meters <https://github.com/bokeh/datashader/blob/master/datashader/utils.py#L142>`_ utility function for working in Web Mercator coordinates with Bokeh
- Added `discussion of why you should be using uniform colormaps <https://anacondausercontent.org/user-content/notebooks/jbednar/plotting_pitfalls?signature=C_divg.WRaRHLPmIEtQ1V1lp0dCBZ34U8Y#6.-Nonuniform-colormapping>`_), and examples of using uniform colormaps from the new `colorcet <https://github.com/bokeh/colorcet>`_ package
- Numerous bug fixes and updates, mostly in the examples and Bokeh extension
- Updated reference manual and documentation

New examples (.ipynb files in examples/):

- `holoviews_datashader <https://anaconda.org/jbednar/holoviews_datashader>`_: Using HoloViews to create dynamic Datashader plots easily
- `census-hv-dask <https://anaconda.org/jbednar/census-hv-dask>`_: Using `GeoViews <https://www.continuum.io/blog/developer-blog/introducing-geoviews>`_ for overlaying shape files, demonstrating gerrymandering by race
- `nyc_taxi-paramnb <https://anaconda.org/jbednar/nyc_taxi-paramnb>`_: Using ParamNB to make a simple dashboard
- `lidar <https://anaconda.org/jbednar/lidar>`_: Visualizing point clouds
- `solar <https://anaconda.org/jbednar/solar>`_: Visualizing solar radiation data
- `Dynamic 1D histogram example <https://anaconda.org/jbednar/nyc_taxi-nongeo>`_ (last code cell in examples/nyc_taxi-nongeo.ipynb)
- dashboard: Now includes opensky example (``python dashboard/dashboard.py -c dashboard/opensky.yml``)

Backwards compatibility:

- To improve consistency with Numpy and Python data structures and eliminate issues with an empty column and row at the edge of the aggregated raster, the provided xrange,yrange bounds are now treated as upper exclusive.  Results will thus differ between 0.5.0 and earlier versions.  See #259 for discussion.

Known issues:

- If you use Jupyter notebook 5.0 (earlier or later versions should be ok), you will need to override a setting that prevents visualizations from appearing, e.g.: ``jupyter notebook --NotebookApp.iopub_data_rate_limit=100000000 census.ipynb &``
- Legend and hover support is currently disabled for the dashboard, due to ongoing development of a simpler approach.

Version 0.4.0 (2016-08-18)
--------------------------

Minor bugfix release to support Bokeh 0.12.1, with some API and defaults changes.

- Added ``examples()`` function to obtain the notebooks and other examples corresponding to the installed datashader version; see `examples/README.md <https://github.com/bokeh/datashader/blob/master/examples/README.md>`_.
- Updated dashboard example to match changes in Bokeh
- Added default color cycle with distinguishable colors for shading categorical data; now ``tf.shade(agg)`` with no other arguments should give a usable plot for both categorical and non-categorical data.

Backwards compatibility:

- Replaced confusing ``tf.interpolate()`` and ``tf.colorize()`` functions with a single shading function ``tf.shade()``. The previous names are still supported, but give deprecation warnings.  Calls to the previous functions using keyword arguments can simply be renamed to use ``tf.shade`` as all the same keywords are accepted, but calls to ``colorize`` that used a positional argument for e.g. the ``color_key`` will now need to use a keyword when calling ``shade()``
- Increased default ``threshold`` for ``tf.dynspread()`` to improve visibility of sparse dots
- Increased default ``min_alpha`` for ``tf.shade()`` (formerly ``tf.colorize()``) to avoid undersaturation

Known issues:

- For Bokeh 0.12.1, some notebooks will give warnings for Bokeh plots when used with Jupyter's "Run All" command.  Bokeh 0.12.2 will fix this problem when it is released, but for now you can either downgrade to 0.12.0 or use single-cell execution.
- There are some Bokeh compatibility issues with the dashboard example that are still being investigated and may require a new Bokeh or datashader release in this series.

Version 0.3.2 (2016-07-18)
--------------------------

Minor bugfix release to support Bokeh 0.12:

- Fixed InteractiveImage zooming to work with Bokeh 0.12.
- Added more responsive event throttling for DynamicImage; ``throttle`` parameter no longer needed and is now deprecated
- Fixed datashader-download-data command
- Improved non-geo Taxi example
- Temporarily disabled dashboard legends; will re-enable in future release

Version 0.3.0 (2016-06-23)
--------------------------

The major feature of this release is support of raster data via ``Canvas.raster``. To use this feature, you must install the optional dependencies via ``conda install rasterio scikit-image``. Rasterio relies on ``gdal`` whose conda package has some known bugs, including a missing dependancy for ``conda install krb5``. InteractiveImage in this release requires bokeh 0.11.1 or earlier, and will not work with bokeh 0.12.

- **PR #160 #187** Improved example notebooks and dashboard
- **PR #186 #184 #178** Add datashader-download-data cli command for grabbing example datasets
- **PR #176 #177** Changed census example data to use HDF5 format (slower but more portable)
- **PR #156 #173 #174** Added Landsat8 and race/ethnicity vs. elevation example notebooks
- **PR #172 #159 #157 #149** Added support for images using ``Canvas.raster`` (requires ``rasterio`` and ``scikit-image``).
- **PR #169** Added legends notebook demonstrating ``create_categorical_legend`` and ``create_ramp_legend`` - **PR #162**. Added notebook example for ``datashader.bokeh_ext.HoverLayer`` - **PR #152**. Added ``alpha``arg to ``tf.interpolate`` - **PR #151 #150, etc.** Small bugfixes
- **PR #146 #145 #144 #143** Added streaming example
- Added ``hold`` decorator to utils, ``summarize_aggregate_values`` helper function
- Added `FAQ <http://datashader.readthedocs.io/en/latest/#faq>`_ to docs

Backwards compatibility:

- Removed ``memoize_method`` -  Renamed ``datashader.callbacks`` --> ``datashader.bokeh_ext`` - Renamed ``examples/plotting_problems.ipynb`` --> ``examples/plotting_pitfalls.ipynb``

Version 0.2.0 (2016-04-01)
--------------------------

A major release with significant new functionality and some small backwards-incompatible changes.

New features:

- **PR #124**, `census <https://anaconda.org/jbednar/census/notebook>`_  New census notebook example, showing how to work with categorical data.
- **PR #79**, `tseries <https://anaconda.org/jbednar/tseries>`_, `trajectory <https://anaconda.org/jbednar/trajectory>`_  Added line glyph and ``.any()``reduction, used in new time series and trajectory notebook examples.
- **PR #76, #77, #131**  Updated all of the other notebooks in examples/, including `nyc_taxi <https://anaconda.org/jbednar/nyc_taxi/notebook>`_.
- **PR #100, #125:** Improved dashboard example: added categorical data support, census and osm datasets, legend and hover support, better performance, out of core option, and more
- **PR #109, #111:** Add full colormap support via a new ``cmap`` argument to ``interpolate`` and ``colorize`` supports color ranges as lists, plus Bokeh palettes and matplotlib colormaps
- **PR #98:** Added ``set_background`` to make it easier to work with images having a different background color than the default white notebooks
- **PR #119, #121:** Added ``eq_hist`` option for ``how`` in interpolate, performing histogram equalization on the data to reveal structure at every intensity level
- **PR #80, #83, #128**: Greatly improved InteractiveImage performance and responsiveness
- **PR #74, #123:** Added operators for spreading pixels (to make individual datapoints visible, as circles, squares, or arbitrary mask shapes) and compositing (for simple and flexible composition of images)

Backwards compatibility:

- The ``low`` and ``high`` color options to ``interpolate`` and ``colorize`` are now deprecated and will be removed in the next release; use ``cmap=[low,high]`` instead.
- The transfer function ``merge`` has been removed to avoid confusion. ``stack`` and others can be used instead, depending on the use case.
- The default ``how`` for ``interpolate`` and ``colorize`` is now ``eq_hist`` to reveal the structure automatically regardless of distribution.
- ``Pipeline`` now has a default ``dynspread`` step, to make isolated points visible when zooming in, and the default sizes have changed.

Version 0.1.0 (2016-04-01)
--------------------------

Initial public release.
