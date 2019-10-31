from datashader.utils import ngjit


@ngjit
def masked_clip_2d(data, mask, lower, upper):
    """
    Clip the elements of an input array between lower and upper bounds,
    skipping over elements that are masked out.

    Parameters
    ----------
    data: np.ndarray
        Numeric ndarray that will be clipped in-place
    mask: np.ndarray
        Boolean ndarray where True values indicate elements that should be
        skipped
    lower: int or float
        Lower bound to clip to
    upper: int or float
        Upper bound to clip to

    Returns
    -------
    None
        data array is modified in-place
    """
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if mask[i, j]:
                continue
            val = data[i, j]
            if val < lower:
                data[i, j] = lower
            elif val > upper:
                data[i, j] = upper
