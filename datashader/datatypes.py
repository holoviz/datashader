import numpy as np
from pandas.api.extensions import (
    ExtensionDtype, ExtensionArray, register_extension_dtype)
from numbers import Integral


@register_extension_dtype
class RaggedDtype(ExtensionDtype):
    name = 'ragged'
    type = np.ndarray
    base = np.dtype('O')

    @classmethod
    def construct_array_type(cls):
        return RaggedArray

    @classmethod
    def construct_from_string(cls, string):
        if string == 'ragged':
            return RaggedDtype()
        else:
            raise TypeError("Cannot construct a '{}' from '{}'"
                            .format(cls, string))


class RaggedArray(ExtensionArray):
    def __init__(self, data, dtype=None):
        """
        Construct a RaggedArray

        Parameters
        ----------
        data
            List or numpy array of lists or numpy arrays
        dtype: np.dtype or str or None (default None)
            Datatype to use to store underlying values from data.
            If none (the default) then dtype will be determined using the
            numpy.result_type function
        """
        if (isinstance(data, dict) and
                all(k in data for k in
                    ['mask', 'start_indices', 'flat_array'])):

            self._mask = data['mask']
            self._start_indices = data['start_indices']
            self._flat_array = data['flat_array']
        else:
            # Compute lengths
            index_len = len(data)
            buffer_len = sum(len(datum)
                             if datum is not None
                             else 0 for datum in data)

            # Compute necessary precision of start_indices array
            for nbits in [8, 16, 32, 64]:
                start_indices_dtype = 'uint' + str(nbits)
                max_supported = np.iinfo(start_indices_dtype).max
                if buffer_len <= max_supported:
                    break

            # infer dtype if not provided
            if dtype is None:
                dtype = np.result_type(*[np.atleast_1d(v)
                                         for v in data
                                         if v is not None])

            # Initialize representation arrays
            self._mask = np.zeros(index_len, dtype='bool')
            self._start_indices = np.zeros(index_len, dtype=start_indices_dtype)
            self._flat_array = np.zeros(buffer_len, dtype=dtype)

            # Populate arrays
            next_start_ind = 0
            for i, array_el in enumerate(data):
                # Check for null values
                isnull = array_el is None

                # Compute element length
                n = len(array_el) if not isnull else 0

                # Update mask
                self._mask[i] = isnull

                # Update start indices
                self._start_indices[i] = next_start_ind

                # Update flat array
                self._flat_array[next_start_ind:next_start_ind+n] = array_el

                # increment next start index
                next_start_ind += n

    @property
    def flat_array(self):
        """
        numpy array containing concatenation of all nested arrays

        Returns
        -------
        np.ndarray
        """
        return self._flat_array

    @property
    def mask(self):
        """
        boolean numpy array the same length as the ragged array where values
        of True indicate missing values.

        Returns
        -------
        np.ndarray
        """
        return self._mask

    @property
    def start_indices(self):
        """
        integer numpy array the same length as the ragged array where values
        represent the index into flat_array where the corresponding ragged
        array element begins.

        Returns
        -------
        np.ndarray
        """
        return self._start_indices

    def __len__(self):
        """
        Length of this array

        Returns
        -------
        length : int
        """
        return len(self._start_indices)

    def __getitem__(self, item):
        """
        Parameters
        ----------
        item : int, slice, or ndarray
            * int: The position in 'self' to get.

            * slice: A slice object, where 'start', 'stop', and 'step' are
              integers or None

            * ndarray: A 1-d boolean NumPy ndarray the same length as 'self'
        """
        if isinstance(item, Integral):
            if item < -len(self) or item >= len(self):
                raise IndexError(item)
            elif self.mask[item]:
                return None
            else:
                # Convert negative item index
                if item < 0:
                    item = 5 + item

                slice_start = self.start_indices[item]
                slice_end = (self.start_indices[item+1]
                             if item + 1 <= len(self) - 1
                             else len(self.flat_array))

                return self.flat_array[slice_start:slice_end]

        elif type(item) == slice:
            data = []
            selected_indices = np.arange(len(self))[item]

            for selected_index in selected_indices:
                data.append(self[selected_index])

            return RaggedArray(data, dtype=self.flat_array.dtype)

        elif isinstance(item, np.ndarray) and item.dtype == 'bool':
            data = []

            for i, m in enumerate(item):
                if m:
                    data.append(self[i])

            return RaggedArray(data, dtype=self.flat_array.dtype)
        else:
            raise KeyError(item)

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):
        """
        Construct a new RaggedArray from a sequence of scalars.

        Parameters
        ----------
        scalars : Sequence
            Each element will be an instance of the scalar type for this
            array, ``cls.dtype.type``.
        dtype : dtype, optional
            Construct for this particular dtype. This should be a Dtype
            compatible with the ExtensionArray.
        copy : boolean, default False
            If True, copy the underlying data.

        Returns
        -------
        RaggedArray
        """
        return RaggedArray(scalars)

    @classmethod
    def _from_factorized(cls, values, original):
        """
        Reconstruct an ExtensionArray after factorization.

        Parameters
        ----------
        values : ndarray
            An integer ndarray with the factorized values.
        original : RaggedArray
            The original ExtensionArray that factorize was called on.

        See Also
        --------
        pandas.factorize
        ExtensionArray.factorize
        """
        return RaggedArray(values, dtype=original.flat_array.dtype)

    def _values_for_factorize(self):
        # Here we return a list of the ragged elements converted into tuples.
        # This is very inefficient, but the elements of this list must be
        # hashable, and we must be able to reconstruct a new Ragged Array
        # from these elements.
        #
        # Perhaps we could replace these tuples with a class that provides a
        # read-only view of an ndarray slice and provides a hash function.
        return [tuple(self[i]) if not self.mask[i] else None
                for i in range(len(self))], None

    def isna(self):
        """
        A 1-D array indicating if each value is missing.

        Returns
        -------
        na_values : np.ndarray
            boolean ndarray the same length as the ragged array where values
            of True represent missing/NA values.
        """
        return self.mask

    def take(self, indices, allow_fill=False, fill_value=None):
        """
        Take elements from an array.

        Parameters
        ----------
        indices : sequence of integers
            Indices to be taken.
        allow_fill : bool, default False
            How to handle negative values in `indices`.

            * False: negative values in `indices` indicate positional indices
              from the right (the default). This is similar to
              :func:`numpy.take`.

            * True: negative values in `indices` indicate
              missing values. These values are set to `fill_value`. Any other
              other negative values raise a ``ValueError``.

        fill_value : any, default None
            Fill value to use for NA-indices when `allow_fill` is True.

        Returns
        -------
        RaggedArray

        Raises
        ------
        IndexError
            When the indices are out of bounds for the array.
        """
        if allow_fill:
            sequence = [self[i] if i >= 0 else fill_value
                        for i in indices]
        else:
            sequence = [self[i] for i in indices]

        return RaggedArray(sequence, dtype=self.flat_array.dtype)

    def copy(self, deep=False):
        """
        Return a copy of the array.

        Parameters
        ----------
        deep : bool, default False
            Also copy the underlying data backing this array.

        Returns
        -------
        RaggedArray
        """
        data = dict(
            mask=self.mask,
            flat_array=self.flat_array,
            start_indices=self.start_indices)

        if deep:
            # Copy underlying numpy arrays
            data = {k: v.copy() for k, v in data.items()}

        return RaggedArray(data)

    @classmethod
    def _concat_same_type(cls, to_concat):
        """
        Concatenate multiple RaggedArray instances

        Parameters
        ----------
        to_concat : list of RaggedArray

        Returns
        -------
        RaggedArray
        """
        # concat masks
        mask = np.hstack(ra.mask for ra in to_concat)

        # concat flat_arrays
        flat_array = np.hstack(ra.flat_array for ra in to_concat)

        # offset and concat start_indices
        offsets = np.hstack([
            [0],
            np.cumsum([len(ra.flat_array) for ra in to_concat[:-1]])])

        start_indices = np.hstack([ra.start_indices + offset
                                   for offset, ra in zip(offsets, to_concat)])

        return RaggedArray(dict(
            mask=mask, flat_array=flat_array, start_indices=start_indices))

    @property
    def dtype(self):
        return RaggedDtype()

    @property
    def nbytes(self):
        """
        The number of bytes needed to store this object in memory.
        """
        return (self._flat_array.nbytes +
                self._start_indices.nbytes +
                self._mask.nbytes)

    def astype(self, dtype, copy=True):
        if isinstance(dtype, RaggedDtype):
            if copy:
                return self.copy()
            return self

        return np.array(self, dtype=dtype, copy=copy)
