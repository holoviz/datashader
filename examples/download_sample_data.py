#!/usr/bin/env python

from __future__ import print_function, absolute_import, division

# -*- coding: utf-8 -*-
"""
Copyright (c) 2011, Kenneth Reitz <me@kennethreitz.com>

Permission to use, copy, modify, and/or distribute this software for any
purpose with or without fee is hereby granted, provided that the above
copyright notice and this permission notice appear in all copies.

THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

clint.textui.progress
~~~~~~~~~~~~~~~~~

This module provides the progressbar functionality.

"""
from collections import OrderedDict
from os import path
import glob
import os
import subprocess
import sys
import tarfile
import time
import zipfile

import yaml
try:
    import requests
except ImportError:
    print('this download script requires the requests module: conda install requests')
    sys.exit(1)

STREAM = sys.stderr

BAR_TEMPLATE = '%s[%s%s] %i/%i - %s\r'
MILL_TEMPLATE = '%s %s %i/%i\r'

DOTS_CHAR = '.'
BAR_FILLED_CHAR = '#'
BAR_EMPTY_CHAR = ' '
MILL_CHARS = ['|', '/', '-', '\\']

# How long to wait before recalculating the ETA
ETA_INTERVAL = 1
# How many intervals (excluding the current one) to calculate the simple moving
# average
ETA_SMA_WINDOW = 9


class Bar(object):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.done()
        return False  # we're not suppressing exceptions

    def __init__(self, label='', width=32, hide=None, empty_char=BAR_EMPTY_CHAR,
                 filled_char=BAR_FILLED_CHAR, expected_size=None, every=1):
        '''Bar is a class for printing the status of downloads'''
        self.label = label
        self.width = width
        self.hide = hide
        # Only show bar in terminals by default (better for piping, logging etc.)
        if hide is None:
            try:
                self.hide = not STREAM.isatty()
            except AttributeError:  # output does not support isatty()
                self.hide = True
        self.empty_char =    empty_char
        self.filled_char =   filled_char
        self.expected_size = expected_size
        self.every =         every
        self.start =         time.time()
        self.ittimes =       []
        self.eta =           0
        self.etadelta =      time.time()
        self.etadisp =       self.format_time(self.eta)
        self.last_progress = 0
        if (self.expected_size):
            self.show(0)

    def show(self, progress, count=None):
        if count is not None:
            self.expected_size = count
        if self.expected_size is None:
            raise Exception("expected_size not initialized")
        self.last_progress = progress
        if (time.time() - self.etadelta) > ETA_INTERVAL:
            self.etadelta = time.time()
            self.ittimes = \
                self.ittimes[-ETA_SMA_WINDOW:] + \
                    [-(self.start - time.time()) / (progress+1)]
            self.eta = \
                sum(self.ittimes) / float(len(self.ittimes)) * \
                (self.expected_size - progress)
            self.etadisp = self.format_time(self.eta)
        x = int(self.width * progress / self.expected_size)
        if not self.hide:
            if ((progress % self.every) == 0 or      # True every "every" updates
                (progress == self.expected_size)):   # And when we're done
                STREAM.write(BAR_TEMPLATE % (
                    self.label, self.filled_char * x,
                    self.empty_char * (self.width - x), progress,
                    self.expected_size, self.etadisp))
                STREAM.flush()

    def done(self):
        self.elapsed = time.time() - self.start
        elapsed_disp = self.format_time(self.elapsed)
        if not self.hide:
            # Print completed bar with elapsed time
            STREAM.write(BAR_TEMPLATE % (
                self.label, self.filled_char * self.width,
                self.empty_char * 0, self.last_progress,
                self.expected_size, elapsed_disp))
            STREAM.write('\n')
            STREAM.flush()

    def format_time(self, seconds):
        return time.strftime('%H:%M:%S', time.gmtime(seconds))


def bar(it, label='', width=32, hide=None, empty_char=BAR_EMPTY_CHAR,
        filled_char=BAR_FILLED_CHAR, expected_size=None, every=1):
    """Progress iterator. Wrap your iterables with it."""

    count = len(it) if expected_size is None else expected_size

    with Bar(label=label, width=width, hide=hide, empty_char=BAR_EMPTY_CHAR,
             filled_char=BAR_FILLED_CHAR, expected_size=count, every=every) \
            as bar:
        for i, item in enumerate(it):
            yield item
            bar.show(i + 1)


def ordered_load(stream, Loader=yaml.Loader, object_pairs_hook=OrderedDict):
    class OrderedLoader(Loader):
        pass
    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))
    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping)
    return yaml.load(stream, OrderedLoader)


class DirectoryContext(object):
    """
    Context Manager for changing directories
    """
    def __init__(self, path):
        self.old_dir = os.getcwd()
        self.new_dir = path

    def __enter__(self):
        os.chdir(self.new_dir)

    def __exit__(self, *args):
        os.chdir(self.old_dir)


def _url_to_binary_write(url, output_path, title):
    '''Given a url, output_path and title,
    write the contents of a requests get operation to
    the url in binary mode and print the title of operation'''
    print('Downloading {0}'.format(title))
    resp = requests.get(url, stream=True)
    try:
        with open(output_path, 'wb') as f:
            total_length = int(resp.headers.get('content-length'))
            for chunk in bar(resp.iter_content(chunk_size=1024), expected_size=(total_length/1024) + 1, every=1000):
                if chunk:
                    f.write(chunk)
                    f.flush()
    except:
        # Don't leave a half-written zip file
        if path.exists(output_path):
            os.remove(output_path)
        raise


def _extract_downloaded_archive(output_path):
    '''Extract a local archive, e.g. zip or tar, then
    delete the archive'''
    if output_path.endswith("tar.gz"):
        with tarfile.open(output_path, "r:gz") as tar:
            tar.extractall()
        os.remove(output_path)
    elif output_path.endswith("tar"):
        with tarfile.open(output_path, "r:") as tar:
            tar.extractall()
        os.remove(output_path)
    elif output_path.endswith("tar.bz2"):
        with tarfile.open(output_path, "r:bz2") as tar:
            tar.extractall()
        os.remove(output_path)
    elif output_path.endswith("zip"):
        with zipfile.ZipFile(output_path, 'r') as zipf:
            zipf.extractall()
        os.remove(output_path)


def _process_dataset(dataset, output_dir, here):
    '''Process each download spec in datasets.yml

    Typically each dataset list entry in the yml has
    "files" and "url" and "title" keys/values to show
    local files that must be present / extracted from
    a decompression of contents downloaded from the url.

    If a url endswith '/', then all files given
    are assumed to be added to the url pattern at the
    end
    '''
    if not path.exists(output_dir):
        os.makedirs(output_dir)

    with DirectoryContext(output_dir) as d:
        requires_download = False
        for f in dataset.get('files', []):
            if not path.exists(f):
                requires_download = True
                break

        if not requires_download:
            print('Skipping {0}'.format(dataset['title']))
            return
        url = dataset['url']
        title_fmt = dataset['title'] + ' {} of {}'
        if url.endswith('/'):
            urls = [url + f for f in dataset['files']]
            output_paths = [os.path.join(here, 'data', fname)
                            for fname in dataset['files']]

            unpacked = ['.'.join(output_path.split('.')[:(-2 if output_path.endswith('gz') else -1)]) + '*'
                        for output_path in output_paths]
        else:
            urls = [url]
            output_paths = [path.split(url)[1]]
            unpacked = dataset['files']
            if not isinstance(unpacked, (tuple, list)):
                unpacked = [unpacked]
        zipped = zip(urls, output_paths, unpacked)
        for idx, (url, output_path, unpack) in enumerate(zipped):
            running_title = title_fmt.format(idx + 1, len(urls))
            if glob.glob(unpack) or os.path.exists(unpack.replace('*','')):
                # Skip a file if a similar one is downloaded:
                # i.e. one that has same name but dif't extension
                print('Skipping {0}'.format(running_title))
                continue
            _url_to_binary_write(url, output_path, running_title)
            _extract_downloaded_archive(output_path)


def main():
    '''Download each dataset specified by datasets.yml in this directory'''
    here = contrib_dir = path.abspath(path.join(path.split(__file__)[0]))
    info_file = path.join(here, 'datasets.yml')
    with open(info_file) as f:
        info = ordered_load(f.read())
        for topic, downloads in info.items():
            output_dir = path.join(here, topic)
            for d in downloads:
                _process_dataset(d, output_dir, here)

if __name__ == '__main__':
    main()
