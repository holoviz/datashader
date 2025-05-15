import re
import sys
from collections import defaultdict
from functools import lru_cache
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
from collections.abc import Callable

_re_no = re.compile(r"\d+")


class VersionError(Exception):
    """Raised when there is a library version mismatch."""

    def __init__(self, msg, version=None, min_version=None, **kwargs):
        self.version = version
        self.min_version = min_version
        super().__init__(msg, **kwargs)


@lru_cache
def _is_installed(module_name):
    return find_spec(module_name) is not None


@lru_cache
def _get_version(package_name):
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "0.0.0"


def _no_import_version(package_name) -> tuple[int, int, int]:
    """Get version number without importing the library"""
    version_str = _get_version(package_name)
    return tuple(map(int, _re_no.findall(version_str)[:3]))


_MIN_SUPPORTED_VERSION = {
    # "pandas": (1, 3, 0),
}

_module_hooks = defaultdict(list)


class _LazyModule:
    def __init__(self, module_name, package_name=None, *, bool_use_sys_modules=False):
        """
        Lazy import module

        This will wait and import the module when an attribute is accessed.

        Parameters
        ----------
        module_name: str
            The import name of the module, e.g. `import PIL`
        package_name: str, optional
            Name of the package, this is the named used for installing the package, e.g. `pip install pillow`.
            Used for the __version__ if the module is not imported.
            If not set uses the module_name.
        bool_use_sys_modules: bool, optional, default False
            Also check `sys.modules` for module in __bool__ check if True.
            This means that bool can only be True if the module is already imported.
        """  # noqa: E501
        self.__module = None
        self.__module_name = module_name
        self.__package_name = package_name or module_name
        self.__bool_use_sys_modules = bool_use_sys_modules
        if module_name in sys.modules:
            self._module

    @property
    def _module(self):
        if self.__module is None:
            self.__module = import_module(self.__module_name)
            if self.__package_name in _MIN_SUPPORTED_VERSION:
                min_version = _MIN_SUPPORTED_VERSION[self.__package_name]
                mod_version = _no_import_version(self.__package_name)
                if mod_version < min_version:
                    min_version_str = ".".join(map(str, min_version))
                    mod_version_str = ".".join(map(str, mod_version))
                    msg = f"{self.__package_name} requires {min_version_str} or higher (found {mod_version_str})"  # noqa: E501
                    raise VersionError(msg, mod_version_str, min_version_str)

            if self in _module_hooks:
                for hook in _module_hooks[self]:
                    hook()
        return self.__module

    def __getattr__(self, attr):
        return getattr(self._module, attr)

    def __dir__(self):
        return dir(self._module)

    def __bool__(self):
        if self.__bool_use_sys_modules:
            return bool(
                self.__module
                or (_is_installed(self.__module_name) and self.__module_name in sys.modules)
            )
        else:
            return bool(self.__module or _is_installed(self.__module_name))

    def __repr__(self):
        if self.__module:
            return repr(self.__module).replace("<module", "<lazy-module")
        else:
            return f"<lazy-module {self.__module_name!r}>"

    @property
    def __version__(self):
        return self.__module and self.__module.__version__ or _get_version(self.__package_name)


def register_import_hook(module: _LazyModule, fn: Callable):
    # NOTE: We cannot use module.__module,
    # because it is overwritten by __getattr__
    if module.__dict__["_LazyModule__module"]:
        fn()
    else:
        _module_hooks[module].append(fn)


TYPE_CHECKING = False
if TYPE_CHECKING:
    import cudf
    import cupy
    import dask
    import dask.array as da
    import dask.bag as db
    import dask.dataframe as dd
    import geopandas as gpd
    import spatialpandas as spd
else:
    cudf = _LazyModule("cudf")
    cupy = _LazyModule("cupy")
    da = _LazyModule("dask.array", "dask")
    dask = _LazyModule("dask")
    dask_cudf = _LazyModule("dask_cudf")
    db = _LazyModule("dask.bag", "dask")
    dd = _LazyModule("dask.dataframe", "dask")
    gpd = _LazyModule("geopandas")
    spd = _LazyModule("spatialpandas")

__all__ = [
    "cudf",
    "cupy",
    "dask",
    "da",
    "db",
    "dd",
    "gpd",
    "spd",
]
