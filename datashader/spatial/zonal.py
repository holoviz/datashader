import numpy as np
import pandas as pd
import warnings
warnings.simplefilter('default')


def zonal_stats(zones, values,
                stat_funcs=['mean', 'max', 'min', 'std', 'var']):
    warnings.warn('\'zonal_stats\' is deprecated. Use \'stats\' instead',
                  DeprecationWarning)

    return stats(zones, values, stat_funcs)


def stats(zones, values, stat_funcs=['mean', 'max', 'min', 'std', 'var']):
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
    stats_df: pandas.DataFrame
        A pandas DataFrame where each column is a statistic
        and each row is a zone with zone id.

    Examples
    --------
    >>> zones_val = np.array([[1, 1, 0, 2],
    >>>                      [0, 2, 1, 2]])
    >>> zones = xarray.DataArray(zones_val)
    >>> values_val = np.array([[2, -1, 5, 3],
    >>>                       [3, np.nan, 20, 10]])
    >>> values = xarray.DataArray(values_val)
    
    # default setting
    >>> df = stats(zones, values)
    >>> df
        mean	max 	min 	std     	var
    1	7.0 	20.0	-1.0	9.273618	86.00
    2	6.5	    10.0   	3.0	    3.500000	12.25

    # custom stat
    >>> custom_stats ={'sum': lambda val: val.sum()}
    >>> df = stats(zones, values)
    >>> df
        sum
    1	21.0
    2	13.0

    """

    if zones.dtype in (np.float32, np.float64):
        zones_val = np.nan_to_num(zones.values).astype(np.int)
    else:
        zones_val = zones.values

    values_val = values.values

    assert zones_val.shape == values_val.shape,\
        "`zones.values` and `values.values` must have same shape"

    assert issubclass(type(zones_val[0, 0]), np.integer),\
        "`zones.values` must be an array of integers"

    assert issubclass(type(values_val[0, 0]), np.integer) or\
           issubclass(type(values_val[0, 0]), np.float),\
        "`values.values` must be an array of integers or floats"

    unique_zones = np.unique(zones_val).astype(int)

    num_zones = len(unique_zones)
    # do not consider zone with 0s
    if 0 in unique_zones:
        num_zones = len(unique_zones) - 1

    # mask out all invalid values_val such as: nan, inf
    masked_values = np.ma.masked_invalid(values_val)

    if isinstance(stat_funcs, dict):
        stats_df = pd.DataFrame(columns=[*stat_funcs])

        for zone_id in unique_zones:
            # do not consider 0 pixels as a zone
            if zone_id == 0:
                continue

            # get zone values_val
            zone_values = np.ma.masked_where(zones_val != zone_id,
                                             masked_values)

            zone_stats = []
            for stat in stat_funcs:
                stat_func = stat_funcs.get(stat)
                if not callable(stat_func):
                    raise ValueError(stat)
                zone_stats.append(stat_func(zone_values))

            stats_df.loc[zone_id] = zone_stats

    else:
        stats_df = pd.DataFrame(columns=stat_funcs)

        for zone_id in unique_zones:
            # do not consider 0 pixels as a zone
            if zone_id == 0:
                continue

            # get zone values_val
            zone_values = np.ma.masked_where(zones_val != zone_id,
                                             masked_values)

            zone_stats = []
            for stat in stat_funcs:
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
                    err_str = 'In function stats(). ' \
                              + '\'' + stat + '\' option not supported.'
                    raise ValueError(err_str)

            stats_df.loc[zone_id] = zone_stats

    num_df_rows = len(stats_df.index)
    assert num_df_rows == num_zones, \
        'Output dataframe must have same number of rows as of zones.values'

    return stats_df


def crosstab(zones, values):
    """Calculate cross-tabulated areas between two datasets: a zone dataset,
    a value dataset (a value raster). Outputs a pandas DataFrame.

    Parameters
    ----------
    zones: xarray.DataArray,
        zones.values is a 2d array of integers.
        A zone is all the cells in a raster that have the same value,
        whether or not they are contiguous. The input zone layer defines
        the shape, values, and locations of the zones. An integer field
        in the zone input is specified to define the zones.

    values: xarray.DataArray,
        values.values is a 2d array of integers or floats.
        The input value raster contains the input values used in calculating
        the categorical statistic for each zone.

    Returns
    -------
    crosstab_df: pandas.DataFrame
        A pandas DataFrame where each column is a pixel value
        and each row is a zone with zone id.

    Examples
    --------
    >>> zones_val = np.array([[1, 1, 0, 2],
    >>>                      [0, 2, 1, 2]])
    >>> zones = xarray.DataArray(zones_val)
    >>> values_val = np.array([[2, -1, 5, 3],
    >>>                       [3, np.nan, 20, 10]])
    >>> values = xarray.DataArray(values_val)
    >>> crosstab_df = crosstab(zones, values)
    >>> crosstab_df
          -1      2 	  3       5       10      20
    1     1       1       0       0       0       1
    2     0       0       1       0       1       0
    """

    # return of the function
    crosstab_df = pd.DataFrame()

    zones_val = zones.values
    values_val = values.values

    assert zones_val.shape == values_val.shape, \
        "`zones.values` and `values.values` must have same shape"

    assert issubclass(type(zones_val[0, 0]), np.integer), \
        "`zones.values` must be an array of integers"

    assert issubclass(type(values_val[0, 0]), np.integer) or \
           issubclass(type(values_val[0, 0]), np.float), \
        "`values.values` must be an array of integers or floats"

    unique_zones = np.unique(zones_val).astype(int)
    num_zones = len(unique_zones)
    # do not consider zone with 0s
    if 0 in unique_zones:
        num_zones = len(unique_zones) - 1

    if num_zones == 0:
        warnings.warn("No zone in `zones` xarray.")

    # mask out all invalid values_val such as: nan, inf
    masked_values = np.ma.masked_invalid(values_val)
    unique_masked = np.unique(masked_values)

    # get unique pixel values (exclude invalid value: nan, inf)
    unique_values = unique_masked[unique_masked.mask == False].data

    if len(unique_values) == 0:
        warnings.warn("No value in `values` xarray.")
        return crosstab_df

    # columns are pixel values
    crosstab_df = pd.DataFrame(columns=unique_values)

    for zone_id in unique_zones:
        # do not consider entries in `zones` with id=0 as a zone
        if zone_id == 0:
            continue

        # get zone values_val
        zone_values = np.ma.masked_where(zones_val != zone_id, masked_values)

        zone_stats = []
        for val in unique_values:
            # count number of `val` pixels in `zone_values`
            count = np.ma.masked_where(zone_values != val, zone_values).count()
            zone_stats.append(count)

        crosstab_df.loc[zone_id] = zone_stats

    num_df_rows = len(crosstab_df.index)
    assert num_df_rows == num_zones, \
        'Output dataframe must have same number of rows as of zones.values'

    return crosstab_df


def apply(zones, agg, func, zone_id):
    """Apply a function to a zone with zone_id. Change the agg content.

    Parameters
    ----------
    zones: xarray.DataArray,
        zones.values is a 2d array of integers.
        A zone is all the cells in a raster that have the same value,
        whether or not they are contiguous. The input zone layer defines
        the shape, values, and locations of the zones. An integer field
        in the zone input is specified to define the zones.

    agg: xarray.DataArray,
        agg.values is a 2d array of integers or floats.
        The input value raster.

    func: callable function to apply to the zone with zone_id

    zone_id: integer

    Returns
    -------

    Examples
    --------
    >>> zones_val = np.array([[1, 1, 0, 2],
    >>>                      [0, 2, 1, 2]])
    >>> zones = xarray.DataArray(zones_val)
    >>> values_val = np.array([[1, 1, 1, 1],
    >>>                       [1, np.nan, 1, 1]])
    >>> agg = xarray.DataArray(values_val)
    >>> func = lambda x: 0
    >>> zone_id = 2
    >>> apply(zones, agg, func, zone_id)
    >>> agg
    >>> array([[1, 1, 1, 0],
    >>>        [1, 0, 1, 0]])
    """

    assert zones.values.shape == agg.values.shape, \
        "`zones.values` and `values.values` must have same shape"

    assert issubclass(type(zones.values[0, 0]), np.integer), \
        "`zones.values` must be an array of integers"

    assert issubclass(type(agg.values[0, 0]), np.integer) or \
           issubclass(type(agg.values[0, 0]), np.float), \
        "`agg.values` must be an array of integers or floats"

    # boolean array to indicate if an entry is in the zone
    zone = zones.values == zone_id
    # apply func to the agg corresponding to the zone
    agg.values[zone] = func(agg.values[zone])
    return

