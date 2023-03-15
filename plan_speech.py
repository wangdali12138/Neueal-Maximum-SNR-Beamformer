import argparse
import json
import librosa as lr
import math
import random as rnd
from pathlib import Path

"""
the file to generate list of used speech files in the directory and store the results in a meta file
"""

parser = argparse.ArgumentParser()
parser.add_argument('--root', default=r'', type=str,
                    help='Root folder with all audio files')
parser.add_argument('--json', default='', type=str, help='JSON with parameters')
parser.add_argument('--speech_meta', default='', type=str, help='Root folder with SPEECH META')
args = parser.parse_args()

with open(args.json, 'r') as f:
    params = json.load(f)


for idx, path in enumerate(Path(args.root).rglob('*.%s' % params['extension'])):
    duration = lr.core.get_duration(filename=path)

    if (duration >= params['duration']):
        for i in range(0, params['repeat']):
            with open(args.speech_meta, mode='a') as f:
                meta = {}
                meta['offset'] = round((duration - params['duration']) * rnd.uniform(0.0, 1.0) * 100) / 100
                meta['duration'] = params['duration']
                meta['path'] = str(path)
                meta_str = json.dumps(meta)
                print(meta_str)
                f.write(meta_str)
                f.write('\n')

