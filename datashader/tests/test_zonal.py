import pytest
import numpy as np
import pandas as pd
import xarray as xa

from datashader.spatial import stats, crosstab, apply


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
    with pytest.raises(Exception) as e_info:
        df = stats(zones=zones, values=values, stat_funcs=custom_stats)


@pytest.mark.stats
def test_stats_invalid_stat_list():
    custom_stats = ['some_stat']
    with pytest.raises(Exception) as e_info:
        df = stats(zones=zones, values=values, stat_funcs=custom_stats)


@pytest.mark.stats
def test_stats_invalid_zones():
    zones = xa.DataArray(np.array([1, 2, 0.5]))
    values = xa.DataArray(np.array([1, 2, 0.5]))

    with pytest.raises(Exception) as e_info:
        df = stats(zones=zones, values=values)


@pytest.mark.stats
def test_stats_invalid_values():
    zones = xa.DataArray(np.array([1, 2, 0], dtype=np.int))
    values = xa.DataArray(np.array(['apples', 'foobar', 'cowboy']))

    with pytest.raises(Exception) as e_info:
        df = stats(zones=zones, values=values)


@pytest.mark.stats
def test_stats_mismatch_zones_values_shape():
    zones = xa.DataArray(np.array([1, 2, 0]))
    values = xa.DataArray(np.array([1, 2, 0, np.nan]))

    with pytest.raises(Exception) as e_info:
        df = stats(zones=zones, values=values)


# test crosstab
@pytest.mark.crosstab
def test_crosstab_invalid_zones():
    zones = xa.DataArray(np.array([1, 2, 0.5]))
    values = xa.DataArray(np.array([1, 2, 0.5]))

    with pytest.raises(Exception) as e_info:
        df = crosstab(zones=zones, values=values)


@pytest.mark.crosstab
def test_crosstab_invalid_values():
    zones = xa.DataArray(np.array([1, 2, 0], dtype=np.int))
    values = xa.DataArray(np.array(['apples', 'foobar', 'cowboy']))

    with pytest.raises(Exception) as e_info:
        df = crosstab(zones=zones, values=values)


@pytest.mark.crosstab
def test_crosstab_mismatch_zones_values_shape():
    zones = xa.DataArray(np.array([1, 2, 0]))
    values = xa.DataArray(np.array([1, 2, 0, np.nan]))

    with pytest.raises(Exception) as e_info:
        df = crosstab(zones=zones, values=values)


# test case 1: no zones
@pytest.mark.crosstab
def test_crosstab_no_zones():
    outline = np.array([[0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0]])
    outline_zones = xa.DataArray(outline)

    outline_val = np.array([[0, 12, 10],
                            [-2.5, np.nan, 7],
                            [np.nan, 3.5, np.inf]])
    # there are 6 values in total" -2.5, 0, 3.5, 7, 10, 12
    num_values = 6

    values = xa.DataArray(outline_val)

    df = crosstab(zones=outline_zones, values=values)

    assert len(df.columns) == num_values


# test case 2: no values
@pytest.mark.crosstab
def test_crosstab_no_zones():
    zones_val = np.array([[0, 1, 3],
                          [0, 1, 4],
                          [0, 0, 2]])
    zones = xa.DataArray(zones_val)

    values_val = np.array([[np.nan, np.nan, np.nan],
                           [np.nan, np.nan, np.nan],
                           [np.nan, np.inf, -np.inf]])
    values = xa.DataArray(values_val)

    df = crosstab(zones=zones, values=values)
    assert len(df) == 0


# test case 3: each zone only has a value
@pytest.mark.crosstab
def test_crosstab():
    zones_val = np.array([[0, 1, 2],
                          [3, 4, 5],
                          [6, 7, 8]])
    zones = xa.DataArray(zones_val)

    values_val = np.array([[0, 1, 2],
                           [3, 4, 5],
                           [6, 7, 8]])
    values = xa.DataArray(values_val)

    df = crosstab(zones=zones, values=values)

    num_rows = len(df.index)
    num_cols = len(df.columns)

    for i in range(num_cols):
        for j in range(1, num_rows + 1):
            if i == j:
                assert df[i][j] == 1
            else:
                assert df[i][j] == 0


# test apply
@pytest.mark.apply
def test_apply_invalid_zones():
    zones = xa.DataArray(np.array([1, 2, 0.5]))
    agg = xa.DataArray(np.array([1, 2, 0.5]))

    with pytest.raises(Exception) as e_info:
        df = apply(zones=zones, agg=agg)


@pytest.mark.apply
def test_apply_invalid_agg():
    zones = xa.DataArray(np.array([1, 2, 0], dtype=np.int))
    agg = xa.DataArray(np.array(['apples', 'foobar', 'cowboy']))

    with pytest.raises(Exception) as e_info:
        df = apply(zones=zones, agg=agg)


@pytest.mark.apply
def test_apply_mismatch_zones_agg_shape():
    zones = xa.DataArray(np.array([1, 2, 0]))
    agg = xa.DataArray(np.array([1, 2, 0, np.nan]))

    with pytest.raises(Exception) as e_info:
        df = apply(zones=zones, agg=agg)


@pytest.mark.apply
def test_apply():
    zones_val = np.array([[0, 1, 2],
                          [3, 4, 5],
                          [6, 7, 8]])
    zones = xa.DataArray(zones_val)

    agg_val = np.array([[0, 1, 2],
                        [3, 4, 5],
                        [6, 7, 8]])
    agg = xa.DataArray(agg_val)
    agg_copy = agg.copy()

    func = lambda x: 0
    zone_idx = np.unique(zones_val)
    for zone_id in zone_idx:
        apply(zones, agg, func, zone_id)
        # agg.shape remains the same
        assert agg.shape == agg_copy.shape
