import warnings
from contextlib import suppress


def main(args=None):
    with suppress(ImportError):
        import pyct  # noqa: F401

        warnings.warn(
            "The 'pyct' package is deprecated since version 0.19 "
            "and will be removed in version 0.20. For downloading sample datasets, "
            "prefer using 'hvsampledata' (for example: "
            "`hvsampledata.nyc_taxi_remote('pandas')`).",
            category=FutureWarning,
            stacklevel=2,
        )
    try:
        import pyct.cmd
    except ImportError:
        import sys
        from . import _missing_cmd
        print(_missing_cmd())
        sys.exit(1)
    return pyct.cmd.substitute_main('datashader',args=args)

if __name__ == "__main__":
    main()
