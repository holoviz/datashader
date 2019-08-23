import pytest
import numpy as np
import pandas as pd
import xarray as xa

from datashader.spatial import zonal_stats


zones_val = np.array([[0, 1, 1, 2, 4, 0, 0],
                      [0, 0, 1, 1, 2, 1, 4],
                      [4, 2, 2, 4, 4, 4, 0]])
zones = xa.DataArray(zones_val)

values_val = np.array([[0, 12, 10, 2, 3.25, np.nan, np.nan],
                       [0, 0, -11, 4, -2.5, np.nan, 7],
                       [np.nan, 3.5, -9, 4, 2, 0, np.inf]])
values = xa.DataArray(values_val)

num_zones = 3
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


@pytest.mark.zonal_stats
def test_zonal_stats_default():
    # default stats=['mean', 'max', 'min', 'std', 'var']
    df = zonal_stats(zones=zones, values=values)

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


@pytest.mark.zonal_stats
def test_zonal_stats_custom_stat():
    cal_sum = lambda values: values.sum()

    def cal_double_sum(values):
        return values.sum() * 2

    zone_sums = [cal_sum(zone_vals_1), cal_sum(zone_vals_2),
                 cal_sum(zone_vals_3)]

    zone_double_sums = [cal_double_sum(zone_vals_1),
                        cal_double_sum(zone_vals_2),
                        cal_double_sum(zone_vals_3)]

    stats = {'sum': cal_sum, 'double sum': cal_double_sum}
    df = zonal_stats(zones=zones, values=values, stats=stats)

    assert isinstance(df, pd.DataFrame)

    # indices of the output DataFrame matches the unique values in `zones`
    idx = df.index.tolist()
    assert idx == unique_values

    num_cols = len(df.columns)
    # there are 2 statistics
    assert num_cols == 2

    assert zone_sums == df['sum'].tolist()
    assert zone_double_sums == df['double sum'].tolist()


@pytest.mark.zonal_stats
def test_zonal_stats_invalid_custom_stat():

    cal_sum = lambda values, zones: values + zones
    stats = {'sum': cal_sum}

    # custom stat only takes 1 argument. Thus, raise error
    with pytest.raises(Exception) as e_info:
        zonal_stats(zones=zones, values=values, stats=stats)
        assert e_info


@pytest.mark.zonal_stats
def test_zonal_stats_invalid_stat_list():
    stats = ['some_stat']
    with pytest.raises(Exception) as e_info:
        zonal_stats(zones=zones, values=values, stats=stats)
        assert e_info


@pytest.mark.zonal_stats
def test_zonal_stats_invalid_zones():
    zones = np.array([1, 2, 0.5])
    values = np.array([1, 2, 0.5])

    with pytest.raises(Exception) as e_info:
        zonal_stats(zones=zones, values=values)
        assert e_info


@pytest.mark.zonal_stats
def test_zonal_stats_invalid_values():
    zones = np.array([1, 2, 0], dtype=np.int)
    values = np.array(['apples', 'foobar', 'cowboy'])

    with pytest.raises(Exception) as e_info:
        zonal_stats(zones=zones, values=values)
        assert e_info


@pytest.mark.zonal_stats
def test_zonal_stats_mismatch_zones_values_shape():
    zones = np.array([1, 2, 0])
    values = np.array([1, 2, 0, np.nan])

    with pytest.raises(Exception) as e_info:
        zonal_stats(zones=zones, values=values)
        assert e_info
