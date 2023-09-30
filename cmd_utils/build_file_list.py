import argparse
import soundfile as sf
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, required=True)
parser.add_argument('--ext', type=str, required=True)
parser.add_argument('--out', type=str, required=True)
parser.add_argument('--sr', type=int, required=False, default=0)

args = parser.parse_args()

file_list = sorted(sum([[
    str(f)
    for f in Path(
        pth).rglob('*.{}'.format(args.ext))] for pth in args.dir.split(',')], []))

if args.sr != 0:
    kept_file_list = []
    for fp in file_list:
        with sf.SoundFile(fp) as sf_obj:
            if sf_obj.samplerate == args.sr:
                kept_file_list.append(fp)

with open(args.out, 'w') as f_out:
    f_out.write('\n'.join(kept_file_list) + '\n')
