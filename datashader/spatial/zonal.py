import numpy as np
import pandas as pd


def zonal_stats(zones, values, stats=['mean', 'max', 'min', 'std', 'var']):
    """Calculate statistics for each zone defined by a zone dataset, based on
    values from another dataset (value raster).

    A single output value is computed for each zone in the input zone dataset.

    Parameters
    ----------
    zones: xarray.DataArray,
        Zone are defined by cells that have the same value,
        whether or not they are contiguous. The input zone layer defines
        the shape, values, and locations of the zones. An integer field
        in the zone input is specified to define the zones.

    values: xarray.DataArray,
        values represent the value raster to be summarized as either integer or float.
        The value raster contains the input values used in calculating
        the output statistic for each zone.

    stats: list of strings or dictionary<stat_name: function(zone_values)>.
        Which statistics to calculate for each zone.
        If a list, possible choices are subsets of
            ['mean', 'max', 'min', 'std', 'var']
        In the dictionary case, all of its values must be callable.
            Function takes only one argument that is the zone values.
            The key become the column name in the output DataFrame.

    Returns
    -------
    zonal_stats_df: pandas.DataFrame
        A pandas DataFrame where each column is a statistic
        and each row is a zone with zone id.

    """

    if zones.dtype in (np.float32, np.float64):
        zones_val = np.nan_to_num(zones.values).astype(np.int)
    else:
        zones_val = zones.values

    values_val = values.values

    assert zones_val.shape == values_val.shape,\
        "`zones.values` and `values.values` must have same shape"

    assert issubclass(type(zones_val[0, 0]), np.integer),\
        "`zones.values` must be an array of integer"

    assert issubclass(type(values_val[0, 0]), np.integer) or\
           issubclass(type(values_val[0, 0]), np.float),\
        "`values.values` must be an array of integer or float"

    unique_zones = np.unique(zones_val).astype(int)

    num_zones = len(unique_zones)
    # do not consider zone with 0s
    if 0 in unique_zones:
        num_zones = len(unique_zones) - 1

    # mask out all invalid values_val such as: nan, inf
    masked_values = np.ma.masked_invalid(values_val)

    if isinstance(stats, dict):

        cols = stats.keys()
        zonal_stats_df = pd.DataFrame(columns=list(cols))

        for zone_id in unique_zones:
            # do not consider 0 pixels as a zone
            if zone_id == 0:
                continue

            # get zone values_val
            zone_values = np.ma.masked_where(zones_val != zone_id,
                                             masked_values)

            zone_stats = []
            for stat in stats:
                stat_func = stats.get(stat)
                if not callable(stat_func):
                    raise ValueError(stat)
                zone_stats.append(stat_func(zone_values))

            zonal_stats_df.loc[zone_id] = zone_stats

    else:
        zonal_stats_df = pd.DataFrame(columns=stats)

        for zone_id in unique_zones:
            # do not consider 0 pixels as a zone
            if zone_id == 0:
                continue

            # get zone values_val
            zone_values = np.ma.masked_where(zones_val != zone_id,
                                             masked_values)

            zone_stats = []
            for stat in stats:
                if stat == 'mean':
                    zone_stats.append(zone_values.mean())
                elif stat == 'max':
                    zone_stats.append(zone_values.max())
                elif stat == 'min':
                    zone_stats.append(zone_values.min())
                elif stat == 'std':
                    zone_stats.append(zone_values.std())
                elif stat == 'var':
                    zone_stats.append(zone_values.var())
                else:
                    err_str = 'In function zonal_stats(). '\
                              + '\'' + stat + '\' option not supported.'
                    raise ValueError(err_str)

            zonal_stats_df.loc[zone_id] = zone_stats

    num_df_rows = len(zonal_stats_df.index)
    assert num_df_rows == num_zones,\
        'Output dataframe must have same number of rows as of zones.values'

    return zonal_stats_df
