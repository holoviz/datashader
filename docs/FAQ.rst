***
FAQ
***


**Q:** When should I use datashader?

**A:** Datashader is designed for working with large datasets, for
cases where it is most crucial to faithfully represent the
*distribution* of your data.  datashader can work easily with
extremely large datasets, generating a fixed-size data structure
(regardless of the original number of records) that gets transferred to
your local browser for display.  If you ever find yourself subsampling
your data just so that you can plot it feasibly, or if you are forced
for practical reasons to iterate over chunks of it rather than looking
at all of it at once, then datashader can probably help you.


**Q:** When should I *not* use datashader?

**A:** If you have a very small number of data points (in the hundreds
or thousands) or curves (in the tens or several tens, each with
hundreds or thousands of points), then conventional plotting packages
like `Bokeh`_ may be more suitable.  With conventional browser-based
packages, all of the data points are passed directly to the browser for
display, allowing specific interaction with each curve or point,
including display of metadata, linking to sources, etc.  This approach
offers the most flexibility *per point* or *per curve*, but rapidly
runs into limitations on how much data can be processed by the browser,
and how much can be displayed on screen and resolved by the human
visual system.  If you are not having such problems, i.e., your data is
easily handled by your plotting infrastructure and you can easily see
and work with all your data onscreen already, then you probably don't
need datashader.

.. _`Bokeh`: https://bokeh.pydata.org


**Q:** Is datashader part of bokeh?

**A:** datashader is an independent project, focusing on generating
aggregate arrays and representations of them as images.  Bokeh is a
complementary project, focusing on building browser-based
visualizations and dashboards.  Bokeh (along with other plotting
packages) can display images rendered by datashader, providing axes,
interactive zooming and panning, selection, legends, hover
information, and so on.  Sample bokeh-based plotting code is provided
with datashader, but viewers for maptlotlib are already under
development, and similar code could be developed for any other
plotting package that can display images.  The library can also be
used separately, without any external plotting packages, generating
images that can be displayed directly or saved to disk, or generating
aggregate arrays suitable for further analysis.
