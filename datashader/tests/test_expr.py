from datashader.expr import Canvas, ByPixel, Point, isreal
from blaze import symbol, summary
from datashape import dshape


canvas = Canvas(plot_height=400, plot_width=600)
df = symbol('df', 'var * {x: float64, y: int64, i32: int32, '
                  'i64: int64, f32: float32, f64: float64}')


def test_isreal():
    assert isreal('int32')
    assert isreal('float32')
    assert not isreal('complex64')


def test_dshape():
    agg = canvas.points(df.x, df.y, i64_sum=df.i64.sum())
    assert agg.dshape == dshape('400 * 600 * {i64_sum: ?int64}')
    assert agg.i64_sum.dshape == dshape('400 * 600 * ?int64')

    agg = canvas.points(df.x, df.y,
                        i64=summary(min=df.i64.min(), max=df.i64.max()),
                        f32_mean=df.f32.mean())
    assert agg.dshape == dshape('400 * 600 * {f32_mean: ?float64, '
                                'i64: {max: ?int64, min: ?int64}}')
    assert agg.i64.dshape == dshape("400 * 600 * {max: ?int64, min: ?int64}")
    assert agg.f32_mean.dshape == dshape("400 * 600 * ?float64")


def test_bypixel():
    agg = canvas.points(df.x, df.y, sums=df.i32.sum())
    assert agg._child is df

    expr = df[df.x > 10]
    agg = canvas.points(expr.x, expr.y, sums=expr.i32.sum())
    assert agg._child is expr
