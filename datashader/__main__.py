import argparse
import inspect
import warnings
import os
import distutils.dir_util

from . import __version__

def install_examples(args):
    """Install examples at the supplied path."""
    source = os.path.join(os.path.dirname(__file__),"examples")
    path = os.path.abspath(args.path)
    if os.path.exists(path):
        warnings.warn("Path %s already exists; will not overwrite newer files."%path)
    distutils.dir_util.copy_tree(source, path, verbose=args.verbose)
    print("Installed examples at %s"%path)

def main(args=None):
    parser = argparse.ArgumentParser(description="Datashader commands")
    parser.add_argument('--version', action='version', version='%(prog)s '+__version__)
    
    subparsers = parser.add_subparsers(title='available commands')

    eg_parser = subparsers.add_parser('install_examples', help=inspect.getdoc(install_examples))
    eg_parser.set_defaults(func=install_examples)
    eg_parser.add_argument('--path',type=str,help='where to install examples',default='datashader-examples')
    eg_parser.add_argument('-v', '--verbose', action='count', default=0)
    
    args = parser.parse_args()

    if hasattr(args,'func'):
        args.func(args)
    else:
        parser.error("must supply command to run") 

if __name__ == "__main__":
    main()
