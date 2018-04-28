def main(args=None):
    try:
        import pvutil.cmd
    except ImportError:
        import sys
        from . import _missing_cmd
        print(_missing_cmd())
        sys.exit(1)
    return pvutil.cmd.substitute_main('datashader',args=args)

if __name__ == "__main__":
    main()
