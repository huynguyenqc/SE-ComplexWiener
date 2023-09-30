import argparse
import os
import shutil
import subprocess
import uuid
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, required=True)
parser.add_argument('--ext', type=str, required=True)

args = parser.parse_args()

file_list = sorted([
    str(f)
    for f in Path(
        args.dir).rglob('*.{}'.format(args.ext))])

for fp in file_list:
    tmp_out_file = f'/tmp/{uuid.uuid4().hex}.wav'
    subprocess.call([
        'ffmpeg',
        '-y',
        '-i',
        fp,
        '-ac',
        '1',
        '-vn',
        '-acodec',
        'pcm_s16le',
        '-ar',
        '16000',
        tmp_out_file,
        '>/dev/null',
        '2>/dev/null'
    ])
    os.remove(fp)
    shutil.move(tmp_out_file, fp)

# ffmpeg -y -i %s -ac 1 -vn -acodec pcm_s16le -ar 16000 %s >/dev/null 2>/dev/null