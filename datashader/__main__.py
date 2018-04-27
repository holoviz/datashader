import argparse
import inspect

from . import __version__
from .cmd import install_examples, download_data


def main(args=None):
    parser = argparse.ArgumentParser(description="Datashader commands")
    parser.add_argument('--version', action='version', version='%(prog)s '+__version__)
    
    subparsers = parser.add_subparsers(title='available commands')

    eg_parser = subparsers.add_parser('install_examples', help=inspect.getdoc(install_examples))
    eg_parser.set_defaults(func=lambda args: install_examples(args.path, args.include_data, args.verbose))
    eg_parser.add_argument('--path',type=str,help='where to install examples',default='datashader-examples')
    eg_parser.add_argument('--include-data',action='store_true',help='Also download data (see download_data command for more flexibility)')
    eg_parser.add_argument('-v', '--verbose', action='count', default=0)

    d_parser = subparsers.add_parser('download_data', help=inspect.getdoc(download_data))
    d_parser.set_defaults(func=lambda args: download_data(args.path,args.datasets_filename))
    d_parser.add_argument('--path',type=str,help='where to download data',default='datashader-examples')
    d_parser.add_argument('--datasets-filename',type=str,help='something',default='datasets.yml')
    d_parser.add_argument('-v', '--verbose', action='count', default=0)
    
    args = parser.parse_args()

    if hasattr(args,'func'):
        args.func(args)
    else:
        parser.error("must supply command to run") 


if __name__ == "__main__":
    main()
