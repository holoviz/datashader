import numpy as np
import pandas as pd
import xarray as xa
import warnings

warnings.simplefilter('default')


def zonal_stats(zones, values,
                stat_funcs=['mean', 'max', 'min', 'std', 'var']):
    warnings.warn('\'zonal_stats\' is deprecated. Use \'stats\' instead',
                  DeprecationWarning)

    return stats(zones, values, stat_funcs)


def stats(zones, values, stat_funcs=['mean', 'max', 'min', 'std', 'var']):
    """Calculate summary statistics for each zone defined by a zone dataset,
    based on values aggregate.

    A single output value is computed for every zone in the input zone dataset.

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

    zones_val = zones.values
    values_val = values.values

    if zones_val.shape != values_val.shape:
        raise ValueError(
            "`zones` and `values` must have same shape")

    if not issubclass(type(zones_val[0, 0]), np.integer):
        raise ValueError("`zones` must be an array of integers")

    if not (issubclass(type(values_val[0, 0]), np.integer) or
            issubclass(type(values_val[0, 0]), np.float)):
        raise ValueError(
            "`values` must be an array of integers or floats")

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
                    err_str = 'Invalid stat name. ' \
                              + '\'' + stat + '\' option not supported.'
                    raise ValueError(err_str)

            stats_df.loc[zone_id] = zone_stats

    return stats_df


def _crosstab(zones, values, layer):
    zones_val = zones.values
    values_val = values.values

    if zones_val.shape != values_val.shape[:-1]:
        raise ValueError(
            "Incompatible shapes between `zones` and `values`")

    if not issubclass(type(zones_val[0, 0]), np.integer):
        raise ValueError("`zones` must be an array of integers")

    if not issubclass(type(values_val[0, 0, 0]), np.integer) and \
            not issubclass(type(values_val[0, 0, 0]), np.float):
        raise ValueError(
            "`values` must be an array of integers or floats")

    if layer is None:
        cats = values.indexes[values.dims[-1]].values
    else:
        if layer not in values.dims:
            raise ValueError("`layer` does not exist in `values` agg.")
        cats = values[layer].values

    num_cats = len(cats)

    unique_zones = np.unique(zones_val).astype(int)
    num_zones = len(unique_zones)

    # do not consider zone with 0s
    if 0 in unique_zones:
        num_zones = len(unique_zones) - 1

    if num_zones == 0:
        warnings.warn("No zone in `zones` xarray.")

    # mask out all invalid values_val such as: nan, inf
    masked_values = np.ma.masked_invalid(values_val)

    # return of the function
    # columns are categories
    crosstab_df = pd.DataFrame(columns=cats)

    for zone_id in unique_zones:
        # do not consider entries in `zones` with id=0 as a zone
        if zone_id == 0:
            continue

        # get all entries in zones with zone_id
        zone_entries = zones_val == zone_id
        zones_entries_3d = np.repeat(zone_entries[:, :, np.newaxis],
                                     num_cats, axis=-1)

        zone_values = zones_entries_3d * masked_values
        zone_cat_stats = [np.sum(zone_cat) for zone_cat in zone_values.T]
        sum_zone_cats = sum(zone_cat_stats)

        # percentage of each category over the zone
        crosstab_df.loc[zone_id] = zone_cat_stats / sum_zone_cats

    return crosstab_df


def crosstab(zones_agg, values_agg, layer=None):
    """Calculate cross-tabulated (categorical stats) areas
    between two datasets: a zone dataset, a value dataset (a value raster).
    Outputs a pandas DataFrame.

    Requires a DataArray with a single data dimension, here called the
    "values_agg", indexed using 3D coordinates.

    DataArrays with 3D coordinates are expected to contain values
    distributed over different categories that are indexed by the
    additional coordinate.  Such an array would reduce to the
    2D-coordinate case if collapsed across the categories (e.g. if one
    did ``aggc.sum(dim='cat')`` for a categorical dimension ``cat``).

    Parameters
    ----------
    zones_agg: xarray.DataArray,
        zones.values is a 2d array of integers.
        A zone is all the cells in a raster that have the same value,
        whether or not they are contiguous. The input zone layer defines
        the shape, values, and locations of the zones. An integer field
        in the zone input is specified to define the zones.

    values_agg: xarray.DataArray,
        values.values is a 3d array of integers or floats.
        The input value raster contains the input values used in calculating
        the categorical statistic for each zone.

    layer: string (optional)
        name of the layer inside the `values_agg` DataArray
        for getting the values
    Returns
    -------
    crosstab_df: pandas.DataFrame
        A pandas DataFrame where each column is a categorical value
        and each row is a zone with zone id.
        Each entry presents the percentage of the category over the zone.
    """

    if not isinstance(zones_agg, xa.DataArray):
        raise TypeError("zones_agg must be instance of DataArray")

    if not isinstance(values_agg, xa.DataArray):
        raise TypeError("values_agg must be instance of DataArray")

    if zones_agg.ndim != 2:
        raise ValueError("zones_agg must be 2D")

    if values_agg.ndim == 3:
        return _crosstab(zones_agg, values_agg, layer)
    else:
        raise ValueError("values_agg must use 3D coordinates")


def apply(zones, values, func):
    """Apply a function to the `values` agg within zones in `zones` agg.
     Change the agg content.

    Parameters
    ----------
    zones: xarray.DataArray,
        zones.values is a 2d array of integers.
        A zone is all the cells in a raster that have the same value,
        whether or not they are contiguous. The input zone layer defines
        the shape, values, and locations of the zones. An integer field
        in the zone input is specified to define the zones.
    agg: xarray.DataArray,
        agg.values is either a 2D or 3D array of integers or floats.
        The input value raster.
    func: callable function to apply.

    Returns
    -------

    Examples
    --------
    >>> zones_val = np.array([[1, 1, 0, 2],
    >>>                      [0, 2, 1, 2]])
    >>> zones = xarray.DataArray(zones_val)
    >>> values_val = np.array([[2, -1, 5, 3],
    >>>                       [3, np.nan, 20, 10]])
    >>> agg = xarray.DataArray(values_val)
    >>> func = lambda x: 0
    >>> apply(zones, agg, func)
    >>> agg
    >>> array([[0, 0, 5, 0],
    >>>        [3, 0, 0, 0]])
    """

    if not isinstance(zones, xa.DataArray):
        raise TypeError("zones_agg must be instance of DataArray")

    if not isinstance(values, xa.DataArray):
        raise TypeError("values_agg must be instance of DataArray")

    if zones.ndim != 2:
        raise ValueError("zones_agg must be 2D")

    if values.ndim != 2 and values.ndim != 3:
        raise ValueError("values_agg must be either 2D or 3D coordinates")

    # get the value of aggs
    zones_val = zones.values
    values_val = values.values

    if zones_val.shape != values_val.shape[:2]:
        raise ValueError(
            "Incompatible shapes between `zones` and `values`")

    if not issubclass(zones.values.dtype.type, np.integer):
        raise ValueError("`zones.values` must be an array of integers")

    if not (issubclass(values.values.dtype.type, np.integer) or
            issubclass(values.values.dtype.type, np.float)):
        raise ValueError(
            "`values` must be an array of integers or float")

    # entries of zone 0 remain the same
    remain_entries = zones_val == 0

    # entries with a non-zero zone value
    zones_entries = zones_val != 0

    if len(values.shape) == 3:
        z = values.shape[-1]
        # add new z-dimension in case 3D `values` aggregate
        remain_entries = np.repeat(remain_entries[:, :, np.newaxis], z,
                                   axis=-1)
        zones_entries = np.repeat(zones_entries[:, :, np.newaxis], z, axis=-1)

    remain_mask = np.ma.masked_array(values_val, mask=remain_entries)
    zones_mask = np.ma.masked_array(values_val, mask=zones_entries)

    # apply func to corresponding `values` of `zones`
    vfunc = np.vectorize(func)
    values_func = vfunc(zones_mask)
    values.values = remain_mask.data * remain_mask.mask \
        + values_func.data * values_func.mask
