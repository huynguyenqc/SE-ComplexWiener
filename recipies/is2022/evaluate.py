import argparse
import concurrent.futures
import csv
import json
import math
import numpy as np
import os
import pesq
import pystoi
import torch
import tqdm
import yaml
from scipy.io import wavfile as sio_wav
from typing import Union, Type, List, Tuple, Dict, Any
from deep.utils import load_state_dict_from_path, AverageLogs
from recipies.is2022.model import SpeechEnhancement, SpeechEnhancementVAE
from recipies.is2022.data_utils import load_wav, tailor_dB_FS, PreSyntheticNoisyDataset


ENHANCE_SAMPLING_RATE = 16000


def si_sdr_dB(x_t: np.ndarray, y_t: np.ndarray) -> float:
    eps = 1e-12
    alpha = (y_t * x_t).mean() / (np.square(x_t).mean() + eps)
    x_t = alpha * x_t
    sisdr_value = float(np.square(x_t).mean() / (np.square(y_t - x_t).mean() + eps))
    sisdr_dB_value = 10 * math.log10(sisdr_value + eps)
    return sisdr_dB_value

    
def inference_worker(
        rank: int,
        audio_dir: str,
        model_class: Union[Type[SpeechEnhancement], Type[SpeechEnhancementVAE]],
        model_configs: Dict[str, Any],
        state_dict_path: str,
        noisy_dataset_list: List[Tuple[str, str]],
        index_list: List[int],
        phase_correction: bool) -> List[Dict[str, Any]]:
    
    model_configs = model_class.ConstructorArgs(**model_configs['model'])
    model = model_class(**model_configs.dict())
    model_state_dict, _, _, _ = load_state_dict_from_path(state_dict_path=state_dict_path)
    model.load_state_dict(model_state_dict)
    model.cuda()
    model.eval()

    loop_generator = (
        tqdm.tqdm(range(len(noisy_dataset_list))) 
        if rank == 0 else 
        range(len(noisy_dataset_list)))

    out_metrics = []
    for i in loop_generator:
        clean_file, noisy_file = noisy_dataset_list[i]
        x_t = load_wav(clean_file, sr=ENHANCE_SAMPLING_RATE)
        y_t = load_wav(noisy_file, sr=ENHANCE_SAMPLING_RATE)

        assert len(x_t) == len(y_t)

        y_t, _, g = tailor_dB_FS(y_t, target_dB_FS=-27.5, eps=1e-12)
        x_t *= g

        y_bt = torch.from_numpy(y_t).unsqueeze_(dim=0).cuda()
        model.eval()
        xHat_bt = model.enhance(y_bt=y_bt, phase_correction=phase_correction)
        xHat_t = xHat_bt.detach().cpu().squeeze_(dim=0).numpy()

        target_len = min(len(x_t), len(xHat_t), len(y_t))
        x_t = x_t[: target_len]
        y_t = y_t[: target_len]
        xHat_t = xHat_t[: target_len]

        metrics_dict = {
            'Index': index_list[i],
            'Clean': clean_file.decode(),
            'Noisy': noisy_file.decode(),
            'Enhanced PESQ-WB': pesq.pesq(fs=ENHANCE_SAMPLING_RATE, ref=x_t, deg=xHat_t, mode='wb'),
            'Enhanced PESQ-NB': pesq.pesq(fs=ENHANCE_SAMPLING_RATE, ref=x_t, deg=xHat_t, mode='nb'),
            'Enhanced STOI': pystoi.stoi(x=x_t, y=xHat_t, fs_sig=ENHANCE_SAMPLING_RATE, extended=False),
            'Enhanced SI-SDR': si_sdr_dB(x_t=x_t, y_t=xHat_t),
            'Noisy PESQ-WB': pesq.pesq(fs=ENHANCE_SAMPLING_RATE, ref=x_t, deg=y_t, mode='wb'),
            'Noisy PESQ-NB': pesq.pesq(fs=ENHANCE_SAMPLING_RATE, ref=x_t, deg=y_t, mode='nb'),
            'Noisy STOI': pystoi.stoi(x=x_t, y=y_t, fs_sig=ENHANCE_SAMPLING_RATE, extended=False),
            'Noisy SI-SDR': si_sdr_dB(x_t=x_t, y_t=y_t)}

        out_metrics.append(metrics_dict)

        stacked_tc = np.stack((x_t, y_t, xHat_t), axis=-1)
        sio_wav.write(
            filename=os.path.join(audio_dir, f'sample_{index_list[i]}.wav'),
            rate=ENHANCE_SAMPLING_RATE,
            data=(stacked_tc * 32768).astype(np.int16))
    return out_metrics


def dict_keep_keys(in_dict: Dict[str, Any], key_list: List[str]) -> Dict[str, Any]:
    key_set = set(key_list)
    return {
        k: v for k, v in in_dict.items() if k in set(key_set)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type', type=str, required=True)
    parser.add_argument('--model-configs', type=str, required=True)
    parser.add_argument('--model-path', type=str, default='')
    parser.add_argument('--wav-list', type=str, required=True)
    parser.add_argument('--phase-correction', action='store_true')
    parser.add_argument('--out-dir', type=str, default='')

    args = parser.parse_args()

    if args.model_path == '':
        args.model_path = os.path.join(os.path.dirname(args.model_configs), 'checkpoints/epoch_800.pt')
    if args.out_dir == '':
        args.out_dir = os.path.join(
            os.path.dirname(args.model_configs), 
            (os.path.splitext(os.path.basename(args.wav_list))[0] 
             + ('_phase' if args.phase_correction else '')))
        if not os.path.exists(args.out_dir):
            os.makedirs(args.out_dir)

    eval_dataset = PreSyntheticNoisyDataset(
        clean_noisy_path=args.wav_list,
        clean_noisy_limit=None, clean_noisy_offset=None, sr=ENHANCE_SAMPLING_RATE,
        sub_sample_sec=None, target_dB_FS=None, target_dB_FS_floating_value=None,
        f0_data_path=None)

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    
    audio_dir = os.path.join(args.out_dir, 'audios')
    if not os.path.exists(audio_dir):
        os.mkdir(audio_dir)

    out_path = os.path.join(args.out_dir, 'results.csv')
    avg_logs = AverageLogs()

    # Get model class
    if args.model_type == 'VQVAE':
        model_class = SpeechEnhancement
    elif args.model_type == 'VAE':
        model_class = SpeechEnhancementVAE
    else:
        raise KeyError(f'Invalid model type! Must be either "VQVAE" or "VAE"; '
                       f'but "{args.model_type}" found!')

    with open(args.model_configs, 'r') as f:
        model_configs = yaml.safe_load(f)

    with open(out_path, 'w') as f_out:
        csv_dict_writer = csv.DictWriter(
            f_out, 
            fieldnames=[
                'Index', 'Clean', 'Noisy',
                'Noisy PESQ-WB', 'Noisy PESQ-NB', 'Noisy STOI', 'Noisy SI-SDR',
                'Enhanced PESQ-WB', 'Enhanced PESQ-NB', 'Enhanced STOI', 'Enhanced SI-SDR'],
            delimiter=',')
        csv_dict_writer.writeheader()

        n_workers = 8
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures_to_rank = {
                executor.submit(
                    inference_worker,
                    rank,
                    audio_dir,
                    model_class,
                    model_configs,
                    args.model_path,
                    eval_dataset.noisy_dataset_list[rank::n_workers],
                    list(range(rank+1, len(eval_dataset)+1, n_workers)),
                    args.phase_correction): rank
                for rank in range(n_workers)}
            for future in concurrent.futures.as_completed(futures_to_rank):
                rank = futures_to_rank[future]
                try:
                    out_metrics = future.result()
                except Exception as exc:
                    print('Process rank {} generated an exception: {}.'.format(rank, str(exc)))
                else:
                    print('Process rank {} finished without exception.'.format(rank))
                    csv_dict_writer.writerows(out_metrics)
                    for out_item in out_metrics:
                        avg_logs.update(
                            dict_keep_keys(out_item, [
                                'Noisy PESQ-WB', 'Noisy PESQ-NB', 'Noisy STOI', 'Noisy SI-SDR',
                                'Enhanced PESQ-WB', 'Enhanced PESQ-NB', 'Enhanced STOI', 
                                'Enhanced SI-SDR']))
                    
    print(json.dumps(avg_logs.get_logs(), indent=4))


if __name__ == '__main__':
    main()
