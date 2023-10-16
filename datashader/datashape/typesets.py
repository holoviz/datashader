"""
Traits constituting sets of types.
"""

from itertools import chain

from .coretypes import (Unit, int8, int16, int32, int64, uint8, uint16, uint32,
                        uint64, float16, float32, float64, complex64,
                        complex128, bool_, Decimal, TimeDelta, Option)


__all__ = ['TypeSet', 'matches_typeset', 'signed', 'unsigned', 'integral',
           'floating', 'complexes', 'boolean', 'numeric', 'scalar',
           'maxtype']


class TypeSet(Unit):
    """
    Create a new set of types. Keyword argument 'name' may create a registered
    typeset for use in datashape type strings.
    """
    __slots__ = '_order', 'name'

    def __init__(self, *args, **kwds):
        self._order = args
        self.name = kwds.get('name')
        if self.name:
            register_typeset(self.name, self)

    @property
    def _set(self):
        return set(self._order)

    @property
    def types(self):
        return self._order

    def __eq__(self, other):
        return (isinstance(other, type(self)) and
                self.name == other.name and self.types == other.types)

    def __hash__(self):
        return hash((self.name, self.types))

    def __contains__(self, val):
        return val in self._set

    def __repr__(self):
        if self.name:
            return '{%s}' % (self.name,)
        return "%s(%s, name=%s)" % (self.__class__.__name__, self._set,
                                    self.name)

    def __or__(self, other):
        return TypeSet(*chain(self, other))

    def __iter__(self):
        return iter(self._order)

    def __len__(self):
        return len(self._set)


def matches_typeset(types, signature):
    """Match argument types to the parameter types of a signature

    >>> matches_typeset(int32, integral)
    True
    >>> matches_typeset(float32, integral)
    False
    >>> matches_typeset(integral, real)
    True
    """
    if types in signature:
        return True
    match = True
    for a, b in zip(types, signature):
        check = isinstance(b, TypeSet)
        if check and (a not in b) or (not check and a != b):
            match = False
            break
    return match


class TypesetRegistry:
    def __init__(self):
        self.registry = {}
        self.lookup = self.registry.get

    def register_typeset(self, name, typeset):
        if name in self.registry:
            raise TypeError("TypeSet %s already defined with types %s" %
                            (name, self.registry[name].types))
        self.registry[name] = typeset
        return typeset

    def __getitem__(self, key):
        value = self.lookup(key)
        if value is None:
            raise KeyError(key)
        return value

registry = TypesetRegistry()
register_typeset = registry.register_typeset
lookup = registry.lookup

#------------------------------------------------------------------------
# Default Type Sets
#------------------------------------------------------------------------

signed = TypeSet(int8, int16, int32, int64, name='signed')
unsigned = TypeSet(uint8, uint16, uint32, uint64, name='unsigned')
integral = TypeSet(*[x for t in zip(signed, unsigned) for x in t],
                   name='integral')
floating = TypeSet(float16, float32, float64, name='floating')
complexes = TypeSet(complex64, complex128, name='complexes')
boolean = TypeSet(bool_, name='boolean')

real = TypeSet(*integral | floating, name='real')
numeric = TypeSet(*integral | floating | complexes, name='numeric')
scalar = TypeSet(*boolean | numeric, name='scalar')


supertype_map = {
    int8: signed,
    int16: signed,
    int32: signed,
    int64: signed,
    uint8: unsigned,
    uint16: unsigned,
    uint32: unsigned,
    uint64: unsigned,
    float16: floating,
    float32: floating,
    float64: floating,
    complex64: complexes,
    complex128: complexes,
    bool_: boolean
}


def supertype(measure):
    """Get the super type of a concrete numeric type

    Examples
    --------
    >>> supertype(int8)
    {signed}

    >>> supertype(float32)
    {floating}

    >>> supertype(complex128)
    {complexes}

    >>> supertype(bool_)
    {boolean}

    >>> supertype(Option(bool_))
    {boolean}
    """
    if isinstance(measure, Option):
        measure = measure.ty
    assert matches_typeset(measure, scalar), 'measure must be numeric'
    return supertype_map[measure]


def maxtype(measure):
    """Get the maximum width for a particular numeric type

    Examples
    --------
    >>> maxtype(int8)
    ctype("int64")

    >>> maxtype(Option(float64))
    Option(ty=ctype("float64"))

    >>> maxtype(bool_)
    ctype("bool")

    >>> maxtype(Decimal(11, 2))
    Decimal(precision=11, scale=2)

    >>> maxtype(Option(Decimal(11, 2)))
    Option(ty=Decimal(precision=11, scale=2))

    >>> maxtype(TimeDelta(unit='ms'))
    TimeDelta(unit='ms')

    >>> maxtype(Option(TimeDelta(unit='ms')))
    Option(ty=TimeDelta(unit='ms'))
    """
    measure = measure.measure
    isoption = isinstance(measure, Option)
    if isoption:
        measure = measure.ty
    if (not matches_typeset(measure, scalar) and
        not isinstance(measure, (Decimal, TimeDelta))):

        raise TypeError('measure must be numeric')

    if measure == bool_:
        result = bool_
    elif isinstance(measure, (Decimal, TimeDelta)):
        result = measure
    else:
        result = max(supertype(measure).types, key=lambda x: x.itemsize)
    return Option(result) if isoption else result
