import argparse

from . import __version__

pvcommands = ['install_examples',
              'download_data']

try:
    from pvutil.cmd import add_pv_commands
except ImportError:
    def add_pv_commands(parser,*args,**kw):
        from . import _install_examples
        for c in pvcommands:
            p = parser.add_parser(c); p.set_defaults(func=lambda args: p.error(_install_examples()))

def main(args=None):
    parser = argparse.ArgumentParser(description="datashader commands")
    parser.add_argument('--version', action='version', version='%(prog)s '+__version__)
    subparsers = parser.add_subparsers(title='available commands')
    add_pv_commands(subparsers,pvcommands,'datashader',args)
    args = parser.parse_args()
    args.func(args) if hasattr(args,'func') else parser.error("must supply command to run") 

if __name__ == "__main__":
    main()
