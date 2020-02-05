import pytest
import numpy as np
import pandas as pd
import xarray as xa

from datashader.spatial.zonal import stats, crosstab, apply

# create valid "zones" and "values" for testing stats()
zones_val = np.array([[0, 1, 1, 2, 4, 0, 0],
                      [0, 0, 1, 1, 2, 1, 4],
                      [4, 2, 2, 4, 4, 4, 0]])
zones = xa.DataArray(zones_val)

values_val = np.array([[0, 12, 10, 2, 3.25, np.nan, np.nan],
                       [0, 0, -11, 4, -2.5, np.nan, 7],
                       [np.nan, 3.5, -9, 4, 2, 0, np.inf]])
values = xa.DataArray(values_val)

unique_values = [1, 2, 4]

masked_values = np.ma.masked_invalid(values.values)

zone_vals_1 = np.ma.masked_where(zones != 1, masked_values)
zone_vals_2 = np.ma.masked_where(zones != 2, masked_values)
zone_vals_3 = np.ma.masked_where(zones != 4, masked_values)

zone_means = [zone_vals_1.mean(), zone_vals_2.mean(), zone_vals_3.mean()]
zone_maxes = [zone_vals_1.max(), zone_vals_2.max(), zone_vals_3.max()]
zone_mins = [zone_vals_1.min(), zone_vals_2.min(), zone_vals_3.min()]
zone_stds = [zone_vals_1.std(), zone_vals_2.std(), zone_vals_3.std()]
zone_vars = [zone_vals_1.var(), zone_vals_2.var(), zone_vals_3.var()]


# --------------------------- TEST stats() ------------------------------------
@pytest.mark.stats
def test_stats_default():
    # default stat_funcs=['mean', 'max', 'min', 'std', 'var']
    df = stats(zones=zones, values=values)

    assert isinstance(df, pd.DataFrame)

    # indices of the output DataFrame matches the unique values in `zones`
    idx = df.index.tolist()
    assert idx == unique_values

    num_cols = len(df.columns)
    # there are 5 statistics in default setting
    assert num_cols == 5

    assert zone_means == df['mean'].tolist()
    assert zone_maxes == df['max'].tolist()
    assert zone_mins == df['min'].tolist()
    assert zone_stds == df['std'].tolist()
    assert zone_vars == df['var'].tolist()


@pytest.mark.stats
def test_stats_custom_stat():
    cal_sum = lambda values: values.sum()

    def cal_double_sum(values):
        return values.sum() * 2

    zone_sums = [cal_sum(zone_vals_1), cal_sum(zone_vals_2),
                 cal_sum(zone_vals_3)]

    zone_double_sums = [cal_double_sum(zone_vals_1),
                        cal_double_sum(zone_vals_2),
                        cal_double_sum(zone_vals_3)]

    custom_stats ={'sum': cal_sum, 'double sum': cal_double_sum}
    df = stats(zones=zones, values=values, stat_funcs=custom_stats)

    assert isinstance(df, pd.DataFrame)

    # indices of the output DataFrame matches the unique values in `zones`
    idx = df.index.tolist()
    assert idx == unique_values

    num_cols = len(df.columns)
    # there are 2 statistics
    assert num_cols == 2

    assert zone_sums == df['sum'].tolist()
    assert zone_double_sums == df['double sum'].tolist()


@pytest.mark.stats
def test_stats_invalid_custom_stat():

    cal_sum = lambda values, zones: values + zones
    custom_stats ={'sum': cal_sum}

    # custom stat only takes 1 argument. Thus, raise error
    with pytest.raises(Exception) as e_info:  # noqa
        stats(zones=zones, values=values, stat_funcs=custom_stats)


@pytest.mark.stats
def test_stats_invalid_stat_list():
    custom_stats = ['some_stat']
    with pytest.raises(Exception) as e_info:  # noqa
        stats(zones=zones, values=values, stat_funcs=custom_stats)


@pytest.mark.stats
def test_stats_invalid_zones():
    zones = xa.DataArray(np.array([1, 2, 0.5]))
    values = xa.DataArray(np.array([1, 2, 0.5]))

    with pytest.raises(Exception) as e_info:  # noqa
        stats(zones=zones, values=values)


@pytest.mark.stats
def test_stats_invalid_values():
    zones = xa.DataArray(np.array([1, 2, 0], dtype=np.int))
    values = xa.DataArray(np.array(['apples', 'foobar', 'cowboy']))

    with pytest.raises(Exception) as e_info:  # noqa
        stats(zones=zones, values=values)


@pytest.mark.stats
def test_stats_mismatch_zones_values_shape():
    zones = xa.DataArray(np.array([1, 2, 0]))
    values = xa.DataArray(np.array([1, 2, 0, np.nan]))

    with pytest.raises(Exception) as e_info:  # noqa
        stats(zones=zones, values=values)


# --------------------------- TEST crosstab() ---------------------------------
@pytest.mark.crosstab
def test_crosstab_invalid_zones():
    # invalid dims (must be 2d)
    zones = xa.DataArray(np.array([1, 2, 0]))

    values = xa.DataArray(np.array([[[1, 2, 0.5]]]),
                          dims=['lat', 'lon', 'race'])
    values['race'] = ['cat1', 'cat2', 'cat3']

    with pytest.raises(Exception) as e_info:
        crosstab(zones_agg=zones, values_agg=values)

    # invalid values (must be int)
    zones = xa.DataArray(np.array([[1, 2, 0.5]]))
    with pytest.raises(Exception) as e_info:  # noqa
        crosstab(zones_agg=zones, values_agg=values)


@pytest.mark.crosstab
def test_crosstab_invalid_values():
    zones = xa.DataArray(np.array([[1, 2, 0]], dtype=np.int))

    # must be either int or float
    values = xa.DataArray(np.array([[['apples', 'foobar', 'cowboy']]]),
                          dims=['lat', 'lon', 'race'])
    values['race'] = ['cat1', 'cat2', 'cat3']

    with pytest.raises(Exception) as e_info:  # noqa
        crosstab(zones_agg=zones, values_agg=values)


@pytest.mark.crosstab
def test_crosstab_mismatch_zones_values_shape():
    zones = xa.DataArray(np.array([[1, 2]]))

    values = xa.DataArray(np.array([[[1, 2, np.nan]]]),
                          dims=['lat', 'lon', 'race'])
    values['race'] = ['cat1', 'cat2', 'cat3']

    with pytest.raises(Exception) as e_info:  # noqa
        crosstab(zones_agg=zones, values_agg=values)


@pytest.mark.crosstab
def test_crosstab_invalid_layer():
    zones = xa.DataArray(np.array([[1, 2]]))

    values = xa.DataArray(np.array([[[1, 2, np.nan]]]),
                          dims=['lat', 'lon', 'race'])
    values['race'] = ['cat1', 'cat2', 'cat3']

    layer = 'cat'
    with pytest.raises(Exception) as e_info:  # noqa
        crosstab(zones_agg=zones, values_agg=values, layer=layer)


# test case 1: no zones
@pytest.mark.crosstab
def test_crosstab_no_zones():
    # create valid `values_agg`
    values_agg = xa.DataArray(np.zeros(24).reshape(2, 3, 4),
                              dims=['lat', 'lon', 'race'])
    values_agg['race'] = ['cat1', 'cat2', 'cat3', 'cat4']
    # create a valid `zones_agg` with compatiable shape
    # no zone
    zones_arr = np.zeros((2, 3), dtype=np.int)
    zones_agg = xa.DataArray(zones_arr)

    num_cats = len(values_agg.dims[-1])
    df = crosstab(zones_agg, values_agg)

    # number of columns = number of categories
    assert len(df.columns) == num_cats
    # no row as no zone
    assert len(df.index) == 0


# test case 2: no values
@pytest.mark.crosstab
def test_crosstab_no_values():
    # create valid `values_agg` of np.nan and np.inf
    values_agg = xa.DataArray(np.zeros(24).reshape(2, 3, 4),
                              dims=['lat', 'lon', 'race'])
    values_agg['race'] = ['cat1', 'cat2', 'cat3', 'cat4']

    # create a valid `zones_agg` with compatiable shape
    zones_arr = np.arange(6, dtype=np.int).reshape(2, 3)
    zones_agg = xa.DataArray(zones_arr)

    df = crosstab(zones_agg, values_agg)

    num_cats = len(values_agg.dims[-1])
    # number of columns = number of categories
    assert len(df.columns) == num_cats

    # exclude region with 0 zone id
    zone_idx = set(np.unique(zones_arr)) - {0}
    num_zones = len(zone_idx)
    # number of rows = number of zones
    assert len(df.index) == num_zones

    num_nans = df.isnull().sum().sum()
    # all are NaN
    assert num_nans == num_zones * num_cats


@pytest.mark.crosstab
def test_crosstab():
    # create valid `values_agg` of np.nan and np.inf
    values_agg = xa.DataArray(np.ones(24).reshape(2, 3, 4),
                              dims=['lat', 'lon', 'race'])
    values_agg['race'] = ['cat1', 'cat2', 'cat3', 'cat4']
    layer = 'race'

    # create a valid `zones_agg` with compatiable shape
    zones_arr = np.arange(6, dtype=np.int).reshape(2, 3)
    zones_agg = xa.DataArray(zones_arr)

    df = crosstab(zones_agg, values_agg, layer)

    num_cats = len(values_agg.dims[-1])
    # number of columns = number of categories
    assert len(df.columns) == num_cats

    # exclude region with 0 zone id
    zone_idx = list(set(np.unique(zones_arr)) - {0})
    num_zones = len(zone_idx)
    # number of rows = number of zones
    assert len(df.index) == num_zones

    num_nans = df.isnull().sum().sum()
    # no NaN
    assert num_nans == 0

    # values_agg are all 1s, so all categories have same percentage over zones
    for col in df.columns:
        assert len(df[col].unique()) == 1

    df['check_sum'] = df.apply(
        lambda r: r['cat1'] + r['cat2'] + r['cat3'] + r['cat4'], axis=1)
    # sum of a row is 1.0
    assert df['check_sum'][zone_idx[0]] == 1.0


# --------------------------- TEST apply() ------------------------------------
@pytest.mark.apply
def test_apply_invalid_agg():
    func = lambda x: 0

    # invalid dims (must be 2d)
    zones = xa.DataArray(np.array([1, 2, 0]))
    values = xa.DataArray(np.array([[[1, 2, 0.5]]]))
    with pytest.raises(Exception) as e_info:
        apply(zones, values, func)

    # invalid zones values (must be int)
    zones = xa.DataArray(np.array([[1, 2, 0.5]]))
    values = xa.DataArray(np.array([[[1, 2, 0.5]]]))
    with pytest.raises(Exception) as e_info:
        apply(zones, values, func)

    zones = xa.DataArray(np.array([[1, 2, 0]]))
    # invalid values (must be int or float)
    values = xa.DataArray(np.array([['apples', 'foobar', 'cowboy']]))
    with pytest.raises(Exception) as e_info:
        apply(zones, values, func)

    zones = xa.DataArray(np.array([[1, 2, 0]]))
    # invalid dim (must be 2d or 3d)
    values = xa.DataArray(np.array([1, 2, 0.5]))
    with pytest.raises(Exception) as e_info:
        apply(zones, values, func)

    zones = xa.DataArray(np.array([[1, 2, 0], [1, 2, 3]]))
    values = xa.DataArray(np.array([[1, 2, 0.5]]))
    # mis-match zones.values.shape and values.values.shape
    with pytest.raises(Exception) as e_info:  # noqa
        apply(zones, values, func)


@pytest.mark.apply
def test_apply():
    func = lambda x: 0

    zones_val = np.zeros((3, 3), dtype=np.int)
    # define some zones
    zones_val[0, ...] = 1
    zones_val[1, ...] = 2
    zones = xa.DataArray(zones_val)

    values_val = np.array([[0, 1, 2],
                           [3, 4, 5],
                           [6, 7, np.nan]])
    values = xa.DataArray(values_val)

    values_copy = values.copy()
    apply(zones, values, func)

    # agg.shape remains the same
    assert values.shape == values_copy.shape

    values_val = values.values
    # values within zones are all 0s
    assert (values_val[0] == [0, 0, 0]).all()
    assert (values_val[1] == [0, 0, 0]).all()
    # values outside zones remain
    assert (values_val[2, :2] == values_copy.values[2, :2]).all()
    # last element of the last row is nan
    assert np.isnan(values_val[2, 2])
