
import os

import requests
import subprocess
try:
    from py7zlib import Archive7z
except:
    subprocess.check_output(['pip', 'install', 'pylzma'], env=os.environ)
    from py7zlib import Archive7z

_d = os.path.dirname(os.path.abspath(__file__))
LIDAR_7Z_DIR = os.path.join(_d, 'data/puget_sound_lidar')
LIDAR_GND_DIR = os.path.join(_d, 'data/puget_sound_lidar')
PUGET_SOUND_LIDAR = 'http://pugetsoundlidar.org/lidardata/newdata/2016/PSLC_KingCounty_Delivery1/BE_ASCII.html'
for f in (LIDAR_GND_DIR, LIDAR_7Z_DIR):
    if not os.path.exists(f):
        os.makedirs(f)


def get_list_of_zips(url=None):

    resp = requests.get(url or PUGET_SOUND_LIDAR)
    content = resp._content.decode('utf-8', 'ignore')
    gen_files = (ci.split('>')[0].strip() for ci in content.split('href='))
    seven_z_files = list(filter(lambda x: '.7z' in x, gen_files))
    urls = [PUGET_SOUND_LIDAR.replace('BE_ASCII.html', z)
            for z in seven_z_files]
    return urls


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
        if not os.path.exists(os.path.dirname(gnd)):
            os.mkdir(os.path.dirname(gnd))
        yield gnd



