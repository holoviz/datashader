
import os

import requests
from py7zlib import Archive7z

_d = os.path.dirname(os.path.abspath(__file__))
LIDAR_7Z_DIR = os.path.join(_d, 'data/puget_sound_lidar')
LIDAR_GND_DIR = os.path.join(_d, 'data/puget_sound_lidar')
PUGET_SOUND_LIDAR = 'http://pugetsoundlidar.org/lidardata/newdata/2016/PSLC_KingCounty_Delivery1/BE_ASCII.html'
for f in (LIDAR_GND_DIR, LIDAR_7Z_DIR):
    if not os.path.exists(f):
        os.makedirs(f)


def get_list_of_zips(url=None):

    c = requests.get(url or PUGET_SOUND_LIDAR)
    content = resp._content.decode('utf-8', 'ignore')
    gen_files = (ci.split('>')[0].strip() for ci in content.split('href='))
    seven_z_files = list(filter(lambda x: '.7z' in x, gen_files))
    return seven_z_files


def download_puget_sound_lidars(seven_z_files=None):
    seven_z_files = seven_z_files or get_list_of_zips()
    for fname in seven_z_files:
        url = PUGET_SOUND_LIDAR.replace('BE_ASCII.html', fname)
        local = os.path.join(LIDAR_7Z_DIR, os.path.basename(fname))
        try:
            with open(local, 'wb') as f:
                content = requests.get(url)._content
        except:
            if os.path.exists(local):
                os.remove(local)
            raise
        f.write(content)
        yield local


def unzip_7z(fname):

    try:
        arc = Archive7z(open(fname, 'rb'))
    except:
        print('FAILED ON 7Z', fname) # q47122e3114.7z has formatting issues
        return
    fnames = arc.filenames
    files = arc.files
    for fn, fi in zip(fnames, files):
        gnd = os.path.join(LIDAR_GND_DIR, os.path.basename(fn))
        yield gnd


def unzip_lidar(lidar_7z_files):
    lidar_gnd_files = []
    lidar_7z_files = download_puget_sound_lidars(seven_z_files=None)
    for fname_7z in lidar_7z_files:
        lidar_gnd_files.extend(unzip_7z(fname_7z))
    return lidar_gnd_files


