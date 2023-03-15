import argparse
import json
import librosa as lr
import math
import numpy as np
import random as rnd
from pathlib import Path


"""
the file to generate list of generated RIRs in the directory, store the results and some value of paramters in a meta file
"""


parser = argparse.ArgumentParser()
parser.add_argument('--root', default='', type=str,
                    help='Root folder with all audio files')
parser.add_argument('--json', default='', type=str, help='JSON with parameters')
parser.add_argument('--farfield_meta', default='', type=str,
                    help='Root folder with FARFIELD META')
args = parser.parse_args()


with open(args.json, 'r') as f:
    params = json.load(f)


for idx, path in enumerate(Path(args.root).rglob('*.%s' % params['extension']['audio'])):
    with open(path.with_suffix('.' + params['extension']['meta']), 'r') as f:
        meta = json.load(f)
    with open(args.farfield_meta, mode='a') as f:
        meta['snr'] = [np.random.uniform(params['snr']['min'], 2), np.random.uniform(-2, params['snr']['max'])]
        meta['gain'] = (np.round(
            np.random.uniform(params['gain']['min'], params['gain']['max'], len(meta['mics'])) * 100) / 100).tolist()
        meta['volume'] = round(rnd.uniform(params['volume']['min'], params['volume']['max']) * 100) / 100
        meta['path'] = str(path)
        meta_str = json.dumps(meta)
        print(meta_str)
        f.write(meta_str)
        f.write('\n')