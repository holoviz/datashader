from contextlib import suppress


def main(args=None):
    with suppress(ImportError):
        import pyct  # noqa: F401

        from . import _warn_pyct_deprecated
        _warn_pyct_deprecated(stacklevel=3)
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
