import pytest

import datashape
from datashape import dshape, has_var_dim, has_ellipsis


def test_cat_dshapes():
    # concatenating 1 dshape is a no-op
    dslist = [dshape('3 * 10 * int32')]
    assert datashape.cat_dshapes(dslist) == dslist[0]
    # two dshapes
    dslist = [dshape('3 * 10 * int32'),
              dshape('7 * 10 * int32')]
    assert datashape.cat_dshapes(dslist) == dshape('10 * 10 * int32')


def test_cat_dshapes_errors():
    # need at least one dshape
    with pytest.raises(ValueError):
        datashape.cat_dshapes([])

    # dshapes need to match after the first dimension
    with pytest.raises(ValueError):
        datashape.cat_dshapes([dshape('3 * 10 * int32'),
                              dshape('3 * 1 * int32')])


@pytest.mark.parametrize('ds_pos',
                         ["... * float32",
                          "A... * float32",
                          "var * float32",
                          "10 * { f0: int32, f1: A... * float32 }",
                          "{ f0 : { g0 : var * int }, f1: int32 }",
                          (dshape("var * int32"),)])
def test_has_var_dim(ds_pos):
    assert has_var_dim(dshape(ds_pos))


@pytest.mark.parametrize('ds_neg',
                         [dshape("float32"),
                          dshape("10 * float32"),
                          dshape("10 * { f0: int32, f1: 10 * float32 }"),
                          dshape("{ f0 : { g0 : 2 * int }, f1: int32 }"),
                          (dshape("int32"),)])
def test_not_has_var_dim(ds_neg):
    assert not has_var_dim(ds_neg)


@pytest.mark.parametrize('ds',
                         [dshape("... * float32"),
                          dshape("A... * float32"),
                          dshape("var * ... * float32"),
                          dshape("(int32, M... * int16) -> var * int8"),
                          dshape("(int32, var * int16) -> ... * int8"),
                          dshape("10 * { f0: int32, f1: A... * float32 }"),
                          dshape("{ f0 : { g0 : ... * int }, f1: int32 }"),
                          (dshape("... * int32"),)])
def test_has_ellipsis(ds):
    assert has_ellipsis(ds)


@pytest.mark.parametrize('ds',
                         [dshape("float32"),
                          dshape("10 * var * float32"),
                          dshape("M * float32"),
                          dshape("(int32, M * int16) -> var * int8"),
                          dshape("(int32, int16) -> var * int8"),
                          dshape("10 * { f0: int32, f1: 10 * float32 }"),
                          dshape("{ f0 : { g0 : 2 * int }, f1: int32 }"),
                          (dshape("M * int32"),)])
def test_not_has_ellipsis(ds):
    assert not has_ellipsis(ds)
