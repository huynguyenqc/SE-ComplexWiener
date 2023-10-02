import numpy as np
import random
import soundfile as sf

from pydantic import BaseModel
from scipy import signal
from torch.utils import data as torch_data
from typing import Any, Dict, List, Optional, Tuple


def load_wav(
        file: str, 
        sr: int = 16000, 
        n_max_samples: Optional[int] = None) -> np.ndarray:

    with sf.SoundFile(file) as aud_file:
        assert aud_file.samplerate == sr, \
            (f'The sampling rate of {file} should be {sr} '
            f'Hz instead of {aud_file.samplerate} Hz!')

        n_samples = aud_file.frames
        if n_max_samples is None:
            s_tc = aud_file.read(
                frames=n_samples, dtype=np.float32,
                always_2d=True)
        else:
            if n_samples > n_max_samples:
                start_idx = np.random.randint(
                    0, n_samples - n_max_samples + 1)
                aud_file.seek(start_idx)
            s_tc = aud_file.read(
                frames=min(n_samples, n_max_samples),
                dtype=np.float32, always_2d=True)
        return s_tc[:, 0]


def norm_amplitude(
        y: np.ndarray,
        scalar: Optional[float] = None,
        eps: float = 1e-6) -> Tuple[np.ndarray, float]:
    if scalar is None:
        scalar = np.amax(np.abs(y)) + eps

    return y / scalar, scalar


def tailor_dB_FS(
        y: np.ndarray,
        target_dB_FS: float = -25,
        eps: float = 1e-6) -> Tuple[np.ndarray, float, float]:
    rms = np.sqrt(np.mean(y ** 2))
    scalar = (10 ** (target_dB_FS / 20)) / (rms + eps)
    y *= scalar
    return y, rms, scalar


def is_clipped(
        y: np.ndarray,
        clipping_threshold: float = 0.999) -> bool:
    return any(np.abs(y) > clipping_threshold)


def subsample(
        data: np.ndarray,
        sub_sample_length: int,
        start_position: int = -1,
        pad_mode: str = 'constant') -> Tuple[np.ndarray, int]:
    """
    Randomly select fixed-length data in last axis
    Args:
        data: to sub-sample data on the last axis
        sub_sample_length: how long
        start_position: If start index smaller than 0,
            randomly generate one index
        return_start_position: Return start position if True
    """
    data_shape = data.shape
    length = data_shape[-1]
    other_dims = data_shape[:-1]

    if length > sub_sample_length:
        if start_position < 0:
            start_position = np.random.randint(length - sub_sample_length)
        end_position = start_position + sub_sample_length
        output_data = data[..., start_position: end_position]
        if end_position > length:
            output_data = np.pad(
                data,
                pad_width=(((0, 0), ) * len(other_dims)
                           + ((0, end_position - length), )),
                mode=pad_mode)
    elif length < sub_sample_length:
        output_data = np.pad(
            data,
            pad_width=(((0, 0), ) * len(other_dims)
                       + ((0, sub_sample_length - length), )),
            mode=pad_mode)
        start_position = 0
    else:
        output_data = data
        start_position = 0

    return output_data, start_position


def offset_and_limit(
        dataset_list: List[Any],
        offset: int = 0,
        limit: Optional[int] = None):
    if limit is None:
        return dataset_list[offset: ]
    return dataset_list[offset: offset + limit]


def parse_snr_range(snr_range: Tuple[float, float]) -> List[float]:
    assert len(snr_range) == 2, \
        f'The range of SNR should be [low, high], not {snr_range}.'
    assert snr_range[0] <= snr_range[-1], \
        f'The low SNR should not larger than high SNR.'

    low, high = snr_range
    snr_list = np.arange(low, high+0.9, 1).tolist()
    return snr_list


def random_select_from(dataset_list: List[Any]) -> Any:
    return random.choice(dataset_list)


def grouping(dataset_list: List[Any], group_size: int) -> List[Any]:
    list_len = len(dataset_list)
    assert list_len % group_size == 0
    return [
        tuple(dataset_list[i: i + group_size]) 
        for i in range(0, list_len, group_size)]


def snr_mix(
        clean_y: np.ndarray,
        noise_y: np.ndarray,
        snr: float,
        target_dB_FS: Optional[float] = None,
        target_dB_FS_floating_value: Optional[float] = None,
        rir: Optional[np.ndarray] = None,
        eps: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray]:
    if rir is not None:
        if rir.ndim > 1:
            rir_idx = np.random.randint(0, rir.shape[0])
            rir = rir[rir_idx, :]

        clean_y = signal.fftconvolve(clean_y, rir)[:len(clean_y)]

    clean_y, _ = norm_amplitude(clean_y)
    clean_y, _, _ = tailor_dB_FS(clean_y, target_dB_FS)
    clean_rms = (clean_y ** 2).mean() ** 0.5

    noise_y, _ = norm_amplitude(noise_y)
    noise_y, _, _ = tailor_dB_FS(noise_y, target_dB_FS)
    noise_rms = (noise_y ** 2).mean() ** 0.5

    snr_a = 10 ** (snr / 20)
    snr_scalar = clean_rms / (noise_rms * snr_a + eps)
    noise_y *= snr_scalar
    noisy_y = clean_y + noise_y

    if target_dB_FS is not None and target_dB_FS_floating_value is not None:
        # Randomly select RMS value of dBFS between -15 dBFS 
        # and -35 dBFS and normalize noisy speech with that value
        noisy_target_dB_FS = np.random.uniform(
            target_dB_FS - target_dB_FS_floating_value,
            target_dB_FS + target_dB_FS_floating_value)

        noisy_y, _, noisy_scalar = tailor_dB_FS(
            noisy_y, noisy_target_dB_FS)
        clean_y *= noisy_scalar

        if is_clipped(noisy_y):
            y_scalar = max(
                np.amax(np.abs(noisy_y)) / (0.99 - eps),
                np.amax(np.abs(clean_y)) / (0.99 - eps))
            noisy_y = noisy_y / y_scalar
            clean_y = clean_y / y_scalar

    return noisy_y, clean_y


class PreSyntheticNoisyDataset(torch_data.Dataset):
    class ConstructorArgs(BaseModel):
        clean_noisy_path: str
        clean_noisy_limit: Optional[int] = None
        clean_noisy_offset: int = 0
        sr: int
        sub_sample_sec: Optional[float] = None
        target_dB_FS: Optional[float] = None
        target_dB_FS_floating_value: Optional[float] = None
        f0_data_path: Optional[str] = None
        dynamic_mixing: bool = False

    def __init__(
            self,
            clean_noisy_path: str,
            clean_noisy_limit: Optional[int],
            clean_noisy_offset: int,
            sr: int,
            sub_sample_sec: Optional[float] = None,
            target_dB_FS: Optional[float] = None,
            target_dB_FS_floating_value: Optional[float] = None,
            f0_data_path: Optional[str] = None,
            dynamic_mixing: bool = False
    ) -> None:
        super(PreSyntheticNoisyDataset, self).__init__()
        self.sr: int = sr
        self.noisy_dataset_list: List[Tuple[str, str]] = offset_and_limit(
            grouping([
                line.rstrip('\n').encode('utf-8')
                for line in open(
                    clean_noisy_path, 'r', encoding='utf-8')],
                group_size=2),
            offset=clean_noisy_offset, limit=clean_noisy_limit)
        self.length: int = len(self.noisy_dataset_list)
        self.sub_sample_sec: Optional[float] = sub_sample_sec
        self.sub_sample_len: Optional[int] = (
            int(sr * sub_sample_sec)
            if sub_sample_sec is not None else None)
        self.target_dB_FS: Optional[float] = target_dB_FS
        self.target_dB_FS_floating_value: Optional[float] = \
            target_dB_FS_floating_value
        self.extract_f0: bool = f0_data_path is not None
        if self.extract_f0:
            with open(f0_data_path, 'r', encoding='utf-8') as f_f0:
                lines = [line.rstrip('\n') for line in f_f0]

                self.pyin_frame_len: int = int(float(lines[0].split('=')[-1]) * sr)
                self.pyin_win_len: int = int(float(lines[1].split('=')[-1]) * sr)
                self.pyin_hop_len: int = int(float(lines[2].split('=')[-1]) * sr)
                self.f0_data_path: Dict[str, str] = {
                    wav_path.encode('utf-8'): f0_path.encode('utf-8') 
                    for wav_path, f0_path in grouping(
                        lines[3: ], group_size=2)}
                self.sub_sample_n_frames: Optional[int] = (
                    (self.sub_sample_len 
                     + self.pyin_frame_len 
                     - self.pyin_hop_len) // self.pyin_hop_len
                    if self.sub_sample_len is not None
                    else None)
        self.dynamic_mixing: bool = dynamic_mixing

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, item: int):
        clean_file, noisy_file = self.noisy_dataset_list[item]
        clean_y = load_wav(clean_file, sr=self.sr)

        if self.dynamic_mixing and self.sub_sample_len is not None and random.random() < 0.8:
            clean_file_2, noisy_file_2 = self.noisy_dataset_list[random.randrange(self.length)]
            clean_y_2 = load_wav(clean_file_2, sr=self.sr)
            noisy_y_2 = load_wav(noisy_file_2, sr=self.sr)
            noise_y = noisy_y_2 - clean_y_2 

            noise_y, _ = subsample(noise_y, self.sub_sample_len, pad_mode='wrap')
            clean_y, start_pos = subsample(clean_y, self.sub_sample_len)

            noisy_y, clean_y = snr_mix(
                clean_y=clean_y, 
                noise_y=noise_y,
                snr=random.uniform(0, 15),
                target_dB_FS=self.target_dB_FS,
                target_dB_FS_floating_value=self.target_dB_FS_floating_value)
        else:
            noisy_y = load_wav(noisy_file, sr=self.sr)

            clean_length = len(clean_y)
            noisy_length = len(noisy_y)
            assert clean_length == noisy_length

            if self.sub_sample_len is not None:
                stacked_y = np.stack((clean_y, noisy_y), axis=0)
                stacked_y, start_pos = subsample(stacked_y, self.sub_sample_len)
                clean_y = stacked_y[0, :]
                noisy_y = stacked_y[1, :]
        
            if (self.target_dB_FS is not None 
                and self.target_dB_FS_floating_value is not None):

                noisy_target_dB_FS = np.random.uniform(
                    self.target_dB_FS - self.target_dB_FS_floating_value,
                    self.target_dB_FS + self.target_dB_FS_floating_value)

                noisy_y, _, noisy_scalar = tailor_dB_FS(noisy_y, noisy_target_dB_FS)
                clean_y *= noisy_scalar

        noisy_y = noisy_y.astype(np.float32)
        clean_y = clean_y.astype(np.float32)

        if self.extract_f0:
            n_frames = (
                clean_y.shape[-1] + self.pyin_frame_len 
                - self.pyin_hop_len) // self.pyin_hop_len
            start_frame_idx = (
                start_pos + self.pyin_frame_len 
                - self.pyin_hop_len) // self.pyin_hop_len
            F0_t = np.load(self.f0_data_path[clean_file])[
                start_frame_idx: start_frame_idx + n_frames]
            if F0_t.shape[-1] < n_frames:
                F0_t = np.pad(
                    F0_t, pad_width=(0, n_frames - F0_t.shape[-1]), 
                    constant_values=np.nan)
            logT0ms_t = -np.log(F0_t / 1000)
            logT0ms_t = logT0ms_t.astype(np.float32)
            return noisy_y, clean_y, logT0ms_t

        return noisy_y, clean_y


class AugmentedNoisyDataset(torch_data.Dataset):
    class ConstructorArgs(BaseModel):
        clean_path: str
        clean_limit: Optional[int] = None
        clean_offset: int = 0
        noise_path: str
        noise_limit: Optional[int] = None
        noise_offset: int = 0
        rir_path: Optional[str] = None
        rir_limit: Optional[int] = None
        rir_offset: Optional[int] = None
        snr_range: Tuple[float, float]
        reverb_proportion: Optional[float]
        silence_sec: float
        target_dB_FS: float
        target_dB_FS_floating_value: float
        sub_sample_sec: float
        sr: int

    def __init__(
        self,
        clean_path: str,
        clean_limit: Optional[int],
        clean_offset: int,
        noise_path: str,
        noise_limit: Optional[int],
        noise_offset: int,
        rir_path: Optional[str],
        rir_limit: Optional[int],
        rir_offset: Optional[int],
        snr_range: Tuple[float, float],
        reverb_proportion: Optional[float],
        silence_sec: float,
        target_dB_FS: float,
        target_dB_FS_floating_value: float,
        sub_sample_sec: float,
        sr: int) -> None:
        """
        Dynamic mixing for training
        Args:
            clean_path: str
            clean_limit: Optional[int]
            clean_offset: int
            noise_path: str
            noise_limit: Optional[int]
            noise_offset: int
            rir_path: str
            rir_limit: Optional[int]
            rir_offset: int
            snr_range: Tuple[float, float]
            reverb_proportion: Optional[float]
            silence_length: float
            target_dB_FS: float
            target_dB_FS_floating_value: float
            sub_sample_sec: float
            sr: int
        """
        super(AugmentedNoisyDataset, self).__init__()
        # acoustics args
        self.sr: int = sr

        self.clean_dataset_list: List[str] = offset_and_limit([
            line.rstrip('\n').encode('utf-8') 
            for line in open(clean_path, 'r', encoding='utf-8')],
            offset=clean_offset, limit=clean_limit)
        self.noise_dataset_list: List[str] = offset_and_limit([
            line.rstrip('\n').encode('utf-8')
            for line in open(noise_path, 'r', encoding='utf-8')],
            offset=noise_offset, limit=noise_limit)

        # random.shuffle(clean_dataset_list)

        if rir_path is not None:
            self.rir_dataset_list: Optional[List[str]] = offset_and_limit([
                line.rstrip('\n').encode('utf-8')
                for line in open(rir_path, 'r')],
                offset=rir_offset, limit=rir_limit)
        else:
            self.rir_dataset_list: Optional[List[str]] = None

        if reverb_proportion is not None:
            assert 0 <= reverb_proportion <= 1, \
                'reverberation proportion should be in [0, 1]'
        self.reverb_proportion: Optional[float] = reverb_proportion

        self.snr_list: List[float] = parse_snr_range(snr_range)
        self.silence_sec: float = silence_sec
        self.silence_length: int = int(silence_sec * sr)
        self.target_dB_FS: float = target_dB_FS
        self.target_dB_FS_floating_value: float = target_dB_FS_floating_value
        self.sub_sample_sec: float = sub_sample_sec
        self.sub_sample_length: int = int(sub_sample_sec * sr)
        self.length: int = len(self.clean_dataset_list)

    def __len__(self) -> int:
        return self.length

    def _select_noise_y(self) -> np.ndarray:
        noise_y = np.zeros(0, dtype=np.float32)
        silence = np.zeros(self.silence_length, dtype=np.float32)
        remaining_length = self.sub_sample_length

        while remaining_length > 0:
            noise_file = random_select_from(self.noise_dataset_list)
            noise_new_added = load_wav(
                noise_file, sr=self.sr, 
                n_max_samples=remaining_length)
            noise_y = np.append(noise_y, noise_new_added)
            remaining_length -= len(noise_new_added)

            # Append silence between noise
            if remaining_length > 0:
                silence_len = min(remaining_length, self.silence_length)
                noise_y = np.append(noise_y, silence[: silence_len])
                remaining_length -= silence_len

        if remaining_length < 0:
            idx_start = np.random.randint(1 - remaining_length)
            noise_y = noise_y[
                idx_start: idx_start + self.sub_sample_length]

        return noise_y

    def __getitem__(self, item: int):
        clean_file = self.clean_dataset_list[item]
        clean_y = load_wav(clean_file, sr=self.sr, 
                           n_max_samples=self.sub_sample_length)
        clean_y, _ = subsample(clean_y, self.sub_sample_length)

        noise_y = self._select_noise_y()
        assert len(clean_y) == len(noise_y), \
            f"Inequality: {len(clean_y)} >< {len(noise_y)}"

        snr = random_select_from(self.snr_list)

        use_reverb = ((
            bool(np.random.random(1) < self.reverb_proportion)
            and (self.rir_dataset_list is not None)) 
            if self.reverb_proportion is not None
            else False)

        noisy_y, clean_y = snr_mix(
            clean_y=clean_y,
            noise_y=noise_y,
            snr=snr,
            target_dB_FS=self.target_dB_FS,
            target_dB_FS_floating_value=self.target_dB_FS_floating_value,
            rir=load_wav(
                random_select_from(self.rir_dataset_list),
                sr=self.sr) if use_reverb else None)

        noisy_y = noisy_y.astype(np.float32)
        clean_y = clean_y.astype(np.float32)

        return noisy_y, clean_y


def main():
    import itertools
    import librosa

    print('# Libri validation set')
    ds = PreSyntheticNoisyDataset(
        clean_noisy_path='./configs/libri_validation.txt',
        clean_noisy_limit=None,
        clean_noisy_offset=0,
        sr=16000)
    for y, x in itertools.islice(ds, 10):
        print(x.shape, y.shape)
    
    print('# Valentini training set')
    ds = PreSyntheticNoisyDataset(
        clean_noisy_path='./configs/valentini_train.txt',
        clean_noisy_limit=None,
        clean_noisy_offset=0,
        sr=16000,
        sub_sample_sec=1.58125,
        target_dB_FS=-27.5,
        target_dB_FS_floating_value=7.5,
        dynamic_mixing=True)

    for y, x in itertools.islice(ds, 10):
        print(x.shape, y.shape, np.abs(x).max(), np.abs(y).max())

    print('# Valentini training set')
    ds = PreSyntheticNoisyDataset(
        clean_noisy_path='./configs/valentini_train.txt',
        clean_noisy_limit=None,
        clean_noisy_offset=0,
        sr=16000,
        sub_sample_sec=1.58125,
        target_dB_FS=-27.5,
        target_dB_FS_floating_value=7.5,
        f0_data_path='./configs/valentini_train_f0.txt')

    print(ds.pyin_frame_len, ds.pyin_win_len, ds.pyin_hop_len, ds.sub_sample_n_frames)
    for k, v in itertools.islice(ds.f0_data_path.items(), 2):
        print(k, v)

    for y, x, logT0ms in itertools.islice(ds, 10):
        f0_true, _, _ = librosa.pyin(
            np.pad(x, (ds.pyin_frame_len - ds.pyin_hop_len,
                       ds.pyin_frame_len - ds.pyin_hop_len)),
            fmin=60., fmax=440., sr=ds.sr,
            frame_length=ds.pyin_frame_len,
            win_length=ds.pyin_win_len,
            hop_length=ds.pyin_hop_len,
            center=False)
        assert f0_true.shape == logT0ms.shape
        print(x.shape, y.shape, logT0ms.shape, np.abs(x).max(), np.abs(y).max())

    print('# Valentini training with augmentation')
    ds = AugmentedNoisyDataset(
        clean_path='./configs/valentini_train_clean_speech.txt',
        clean_limit=None,
        clean_offset=0,
        noise_path='./configs/musan_music_noise.txt',
        noise_limit=None,
        noise_offset=0,
        rir_path=None,
        rir_limit=None,
        rir_offset=0,
        snr_range=(-5., 20.),
        reverb_proportion=0.75,
        silence_sec=0.2,
        target_dB_FS=-25.,
        target_dB_FS_floating_value=10.,
        sub_sample_sec=3.0,
        sr=16000)
    for y, x in itertools.islice(ds, 10):
        print(x.shape, y.shape, np.abs(y).max(), np.abs(x).max())

    print(len(ds) // 256)


if __name__ == '__main__':
    main()