# Datashader Roadmap, as of 2/2017

Datashader is an open-source project, with contributions from a variety of developers with different priorities, so it is not possible to lay out a fully detailed timeline of upcoming features.  That said, there are clear priorities that the current developers have agreed on, which will be described here and updated occasionally. 

If you need any of the functionality listed below and want to help make it a priority, please respond to the relevant issue listed (preferably with offers of coding, financial, or other assistance!). 


1. **Ongoing maintenance, improved documentation and examples**
  - As always, there are various bugs and usability issues reported on the issue tracker, and we will address these as time permits.
  - The user manual is currently minimal, while the real documentation is spread over various notebooks, making it difficult to browse. ([#137](../../issues/137))
  - The notebooks should nearly all be reworked to use HoloViews, to make them simpler and to use a recommended workflow.

2. **More consistent and powerful shading and aggregation**
  - "alpha" (opacity) should be supported for non-categorical shade() just as for categorical aggregates ([#244](../../issues/244)), which will make it simpler to overlay data onto backgrounds.
  - Should be able to aggregate any field categorically, not just counts ([#140](../../issues/140))

3. **Optimizing file formats** ([#129](../../issues/129))

4. **Better integration with external plotting libraries (Bokeh, HoloViews, matplotlib)**
  - Datashader needs to provide functions for supporting hover information, legends, colorbars, and interactivity, which each plotting library can then use ([#126](../../issues/126), [#127](../../issues/127), [#136](../../issues/136), [#251](../../issues/251))
  - HoloViews supports basic operations with aggregate arrays, but will need extensions to be able to handle everything possible in datashader itself.
  - There is a draft of Matplotlib support ([#200](../../issues/200)), but it will need significant work before it is usable for most purposes.
  - HoloViews should be able to allow users to set criteria for when datashader will be substituted for a Points or Path plot, based on size
  - Support for Bokeh server ([#97](../../issues/97), [#147](../../issues/147), [#271](../../issues/271), [HoloViews #694](https://github.com/ioam/holoviews/issues/694))

5. **Large graph/network layout rendering**

6. **Support for rendering surfaces (as orthgraphic projections) from arbitrary samples** ([#181](../../issues/181))

7. **Optimizing data access (via quadtree/kdtree dataset partitioning) and deployment** (as slippy-map tiles [#246](../../issues/246))

8. **Better datetime and timeseries support**
  - Native datetime64 axes not supported (must be numeric); currently must convert to int64 and use HoloViews to format as a date/time string. ([#270](../../issues/270), [#114](../../issues/114))

9. **Visualizing uncertainty, anomalies, stability**
  - Example of plotting points with associated probabilities ([#102](../../issues/102))
  - Tools for anomaly detection ([#116](../../issues/116))
  - Tools for stability analysis ([#115](../../issues/115))

10. **Misc:**
  - [#132](../../issues/132) GPU support
  - [#113](../../issues/113) Seam carving support 
  - [#110](../../issues/110) 1D aggregation example
  - [#105](../../issues/105) Cyclical data example
  - [#103](../../issues/103) Symbolic rendering of aggregate array
  -  [#92](../../issues/92)  Box select support
  -  [#61](../../issues/61)  Add information on requirements for osm example
  -  [#14](../../issues/14)  Changes that appear to require Bokeh extensions
  -  [#13](../../issues/13)  Axis labels (in lat/lon?) for examples like dashboard.py
  - [#242](../../issues/242) Spatiotemporal data animation
  - [#273](../../issues/273) Add LIDAR example
