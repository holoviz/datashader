#!/usr/bin/env python

"""
Convert PCAP output to undirected graph and save in Parquet format.
"""

from __future__ import print_function

import re
import socket
import struct
import sys

import fastparquet as fp
import numpy as np
import pandas as pd


def ip_to_integer(s):
    return struct.unpack("!I", socket.inet_aton(s))[0]


def to_parquet(filename, prefix="maccdc2012"):
    with open(filename) as f:
        traffic = {}
        nodes = set()

        for line in f.readlines():
            fields = line.split()
            if not fields:
                continue
            try:
                addresses = []

                # Extract source IP address and convert to integer
                m = re.match(r'\d+\.\d+\.\d+\.\d+', fields[2])
                if not m:
                    continue
                addresses.append(ip_to_integer(m.group(0)))

                # Extract target IP address and convert to integer
                m = re.match(r'\d+\.\d+\.\d+\.\d+', fields[4])
                if not m:
                    continue
                addresses.append(ip_to_integer(m.group(0)))

                nodes = nodes.union(addresses)
                key = tuple(sorted(addresses))

                # Extract packet size
                nbytes = int(fields[-1])

                if key in traffic:
                    traffic[key] += nbytes
                else:
                    traffic[key] = nbytes
            except:
                pass

        # Anonymize IP addresses by subtracting minimum from all integers
        min_node_id = min(nodes)
        edges = []
        for key in traffic:
            edge = [key[0] - min_node_id, key[1] - min_node_id, traffic[key]]
            edges.append(edge)

        nodes_df = pd.DataFrame(np.array(list(nodes)) - min_node_id, columns=['id'])
        edges_df = pd.DataFrame(np.array(edges), columns=['source', 'target', 'weight'])

        fp.write('{}_nodes.parq'.format(prefix), nodes_df)
        fp.write('{}_edges.parq'.format(prefix), edges_df)

if __name__ == '__main__':
    if len(sys.argv) >= 2:
        to_parquet(sys.argv[1], prefix=sys.argv[2])
    else:
        to_parquet(sys.argv[1])
