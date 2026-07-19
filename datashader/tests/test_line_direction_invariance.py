import numpy as np
import pandas as pd
import pytest
import datashader as ds

gpd = pytest.importorskip("geopandas")
shapely = pytest.importorskip("shapely")

def test_geopandas_line_direction_invariance():
    """
    Test that rendering a single 2-point line segment via the GeoPandas / Shapely LineString 
    code path is direction-invariant (i.e. produces the exact same raster image whether 
    drawn forward or backward).
    
    This verifies that the last endpoint is correctly flagged with segment_end = True, 
    so that an end cap is rendered at the final vertex in both directions.
    """
    # A single horizontal segment well inside the canvas
    # Forward: (2, 5) -> (8, 5)
    gdf_fwd = gpd.GeoDataFrame(geometry=[shapely.LineString([(2.0, 5.0), (8.0, 5.0)])])
    # Backward: (8, 5) -> (2, 5)
    gdf_rev = gpd.GeoDataFrame(geometry=[shapely.LineString([(8.0, 5.0), (2.0, 5.0)])])
    
    cvs = ds.Canvas(plot_width=10, plot_height=10, x_range=(0, 10), y_range=(0, 10))
    
    # Render both using antialiased lines (line_width=1)
    img_fwd = cvs.line(gdf_fwd, geometry="geometry", agg=ds.count(), line_width=1).fillna(0).values
    img_rev = cvs.line(gdf_rev, geometry="geometry", agg=ds.count(), line_width=1).fillna(0).values
    
    # They must be identical
    np.testing.assert_allclose(img_fwd, img_rev, atol=1e-6)


def test_tiled_boundary_invariance():
    """
    Test that rendering a segment clipped by the canvas boundary (representing a tile boundary) 
    is direction-invariant. Without the segment_end propagation fix, one direction (forward) 
    fails to draw a cap at the clipped end while the other (reverse) draws a cap, resulting 
    in a rendering mismatch.
    """
    # A diagonal segment that extends beyond the canvas right boundary (clipped on the right)
    # Forward: (2, 2) -> (15, 8)
    gdf_fwd = gpd.GeoDataFrame(geometry=[shapely.LineString([(2.0, 2.0), (15.0, 8.0)])])
    # Backward: (15, 8) -> (2, 2)
    gdf_rev = gpd.GeoDataFrame(geometry=[shapely.LineString([(15.0, 8.0), (2.0, 2.0)])])
    
    cvs = ds.Canvas(plot_width=10, plot_height=10, x_range=(0, 10), y_range=(0, 10))
    
    img_fwd = cvs.line(gdf_fwd, geometry="geometry", agg=ds.count(), line_width=1).fillna(0).values
    img_rev = cvs.line(gdf_rev, geometry="geometry", agg=ds.count(), line_width=1).fillna(0).values
    
    np.testing.assert_allclose(img_fwd, img_rev, atol=1e-6)
