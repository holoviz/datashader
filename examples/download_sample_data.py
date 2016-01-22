"""Download data needed for the examples"""
from __future__ import print_function
from os import path, makedirs
import pandas as pd
import numpy as np

examples_dir = path.dirname(path.realpath(__file__))
data_dir = path.join(examples_dir, 'data')
if not path.exists(data_dir):
    makedirs(data_dir)

# Taxi data
def latlng_to_meters(df, lat_name, lng_name):
    lat = df[lat_name]
    lng = df[lng_name]
    origin_shift = 2 * np.pi * 6378137 / 2.0
    mx = lng * origin_shift / 180.0
    my = np.log(np.tan((90 + lat) * np.pi / 360.0)) / (np.pi / 180.0)
    my = my * origin_shift / 180.0
    df.loc[:, lng_name] = mx
    df.loc[:, lat_name] = my

taxi_path = path.join(data_dir, 'taxi.csv')
if not path.exists(taxi_path):
    print("Downloading Taxi Data...")
    url = ('https://storage.googleapis.com/tlc-trip-data/2015/'
           'yellow_tripdata_2015-01.csv')
    df = pd.read_csv(url)
    df = df.loc[(df.pickup_longitude < -73.90) &
                (df.pickup_longitude > -74.05) &
                (df.dropoff_longitude < -73.90) &
                (df.dropoff_longitude > -74.05) &
                (df.pickup_latitude > 40.70) &
                (df.pickup_latitude < 40.80) &
                (df.dropoff_latitude > 40.70) &
                (df.dropoff_latitude < 40.80)].copy()
    latlng_to_meters(df, 'pickup_latitude', 'pickup_longitude')
    latlng_to_meters(df, 'dropoff_latitude', 'dropoff_longitude')
    df.to_csv(taxi_path, index=False)

print("\nAll data downloaded.")
