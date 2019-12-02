# Datashader Roadmap, as of 4/2018

Datashader is an open-source project, with contributions from a variety of developers with different priorities, so it is not possible to lay out a fully detailed timeline of upcoming features.  That said, there are clear priorities that the current developers have agreed on, which will be described here and updated occasionally.

If you need any of the functionality listed below and want to help make it a priority, please respond to the relevant issue listed (preferably with offers of coding, financial, or other assistance!).

1. **Ongoing maintenance, improved documentation and examples**
  - As always, there are various bugs and usability issues reported on the issue tracker, and we will address these as time permits.
  - Some of the notebooks and the included dashboard need rework to use HoloViews, to make them simpler and to use a recommended workflow.

2. **Optimizing data access (via quadtree/kdtree dataset partitioning) and deployment** (including as slippy-map tiles [#246](../../issues/246)) [Scheduled for 2018]

3. **Better integration with external plotting libraries (Bokeh, HoloViews, matplotlib)**
  - Datashader needs to provide functions for supporting hover information, legends, colorbars, and interactivity, which each plotting library can then use ([#126](../../issues/126), [#127](../../issues/127), [#136](../../issues/136), [#251](../../issues/251))
  - There is a draft of Matplotlib support ([#200](../../issues/200)), but it will need significant work before it is usable for most purposes.
  - HoloViews should be able to allow users to set criteria for when datashader will be substituted for a Points or Path plot, based on size

4. **More consistent and powerful shading and aggregation**
  - Should be able to aggregate any field categorically, not just counts ([#140](../../issues/140))

5. **Visualizing uncertainty, anomalies, stability**
  - Example of plotting points with associated probabilities ([#102](../../issues/102))
  - Tools for stability analysis ([#115](../../issues/115))

6. **Misc:**
  - [#132](../../issues/132) GPU support
  - [#110](../../issues/110) 1D aggregation example
  - [#105](../../issues/105) Cyclical data example
  - [#103](../../issues/103) Symbolic rendering of aggregate array
  -  [#92](../../issues/92)  Box select support
  -  [#61](../../issues/61)  Add information on requirements for osm example
  - [#242](../../issues/242) Spatiotemporal data animation
