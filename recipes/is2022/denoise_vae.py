import argparse
import contextlib
import datetime
import numpy as np
import os
import shutil
import torch
import yaml
from deep.distributed import init_distributed
from pydantic import BaseModel
from torch import nn, optim
from torch.utils import data as torch_data
from typing import Any, Dict, Generator, List, Optional, Union, Literal

from deep.utils import (
    AverageLogs, InstantLogs, Checkpointer, Counter, CSVLogWriter,
    load_model_to_cuda, load_state_dict_from_path)
from recipes.is2022.data_utils import PreSyntheticNoisyDataset, AugmentedNoisyDataset
from recipes.is2022.model import PretrainSpeechVarianceVAE, SpeechEnhancementVAE
from recipes.is2022.media_logging import MediaDataLogging
from utils.snapshot_src import snapshot


torch.manual_seed(3407)
np.random.seed(3407)


class DataLoaderConfigs(BaseModel):
    batch_size: int = 1
    num_workers: int = 0
    shuffle: Optional[bool] = None
    pin_memory: bool = False
    drop_last: bool = False


class TrainConfigs(BaseModel):
    n_epochs: int
    iter_per_checkpoint: int
    epoch_per_val: int
    epoch_per_checkpoint: int
    optimiser: Dict[str, Any]
    scheduler: Dict[str, Any]
    train_data_type: Literal['PreSynctheticNoisyDataset', 'AugmentedNoisyDataset'] = 'PreSynctheticNoisyDataset'
    train_data: Union[PreSyntheticNoisyDataset.ConstructorArgs, AugmentedNoisyDataset.ConstructorArgs]
    train_dataloader: DataLoaderConfigs
    validation_data: PreSyntheticNoisyDataset.ConstructorArgs
    validation_dataloader: DataLoaderConfigs
    auto_mix_precision: bool = False


class EpochValidator(MediaDataLogging):
    def __init__(
            self,
            dataloader: torch_data.DataLoader,
            model: SpeechEnhancementVAE,
            save_dir: str,
            log_writer: CSVLogWriter,
            epoch_period: int) -> None:
        super(EpochValidator, self).__init__()

        self.model_ref: SpeechEnhancementVAE = model
        self.val_dataloader: torch_data.DataLoader = dataloader
        self.save_dir: str = save_dir
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        self.csv_log_writer: int = log_writer
        self.epoch_period: int = epoch_period

    def validate(self, epoch: int) -> None:
        if (epoch + 1) % self.epoch_period == 0:
            if os.path.exists(os.path.join(self.save_dir, f'epoch_{epoch + 1 - self.epoch_period}')):
                shutil.rmtree(os.path.join(self.save_dir, f'epoch_{epoch + 1 - self.epoch_period}'), ignore_errors=True)

            target_dir = os.path.join(self.save_dir, f'epoch_{epoch + 1}')
            assert not os.path.exists(target_dir)
            figure_target_dir = os.path.join(target_dir, 'figures') 
            waveform_target_dir = os.path.join(target_dir, 'waveform') 
            os.mkdir(target_dir)
            os.mkdir(figure_target_dir)
            os.mkdir(waveform_target_dir)

            logs = AverageLogs()

            for i, batch_data in enumerate(self.val_dataloader):
                y_bt = batch_data[0].cuda()
                x_bt = batch_data[1].cuda()

                self.model_ref.eval()
                val_data = self.model_ref.validate(x_bt, y_bt, epoch=epoch)

                # Write numerical values to CSV
                if val_data.get('numerical') is not None:
                    logs.update(val_data['numerical'])

                # Plot spectrum and mask (images)
                if val_data.get('spectrum') is not None:
                    self.plot_dict_spectrum(val_data['spectrum'], figure_target_dir, i)
                if val_data.get('mask') is not None:
                    self.plot_dict_mask(val_data['mask'], figure_target_dir, i)

                # Write wav files
                if val_data.get('waveform') is not None:
                    self.write_dict_wav(val_data['waveform'], waveform_target_dir, i)

            self.csv_log_writer.write_log(logs.get_logs())


class TrainContext:
    def __init__(
            self,
            log_dir: str,
            configs: Dict[str, Any],
            field_names: List[str],
            model: nn.Module,
            optimiser: optim.Optimizer,
            scheduler: Optional[optim.lr_scheduler.LRScheduler],
            train_dataloader: torch_data.DataLoader,
            val_dataloader: torch_data.DataLoader,
            iter_save_period: int,
            epoch_val_period: int,
            epoch_save_period: int) -> None:
        
        self.log_dir: str = log_dir
        self.configs: Dict[str, Any] = configs
        self.field_names: List[str] = field_names
        self.model: nn.Module = model
        self.optimiser: optim.Optimizer = optimiser
        self.scheduler: Optional[optim.lr_scheduler.LRScheduler] = scheduler
        self.train_dataloader: torch_data.DataLoader = train_dataloader
        self.val_dataloader: torch_data.DataLoader = val_dataloader
        self.iter_save_period: int = iter_save_period
        self.epoch_val_period: int = epoch_val_period
        self.epoch_save_period: int = epoch_save_period

        self.epoch_train_log_writer: Optional[CSVLogWriter] = None
        self.iter_train_log_writer: Optional[CSVLogWriter] = None
        self.iter_val_log_writer: Optional[CSVLogWriter] = None

        self.checkpointer: Optional[Checkpointer] = None
        self.validator: Optional[EpochValidator] = None

        self.iter_counter: Optional[Counter] = None

    def start_context(self):
        global rank
        if rank == 0:

            # Create directory
            assert not os.path.exists(self.log_dir), 'The folder has already existed!'
            os.makedirs(self.log_dir)

            with open(os.path.join(self.log_dir, 'configs.yml'), 'w') as f_configs:
                yaml.safe_dump(data=self.configs, stream=f_configs, default_flow_style=False)

            # Snapshot source code before training
            snapshot_src_dir = os.path.join(self.log_dir, 'src')
            os.mkdir(snapshot_src_dir)
            snapshot(
                snapshot_src_dir, 
                source_packages = [
                    'deep', 'recipies/is2022', 'utils', 'sample_data',
                    'run.sh', 'requirements.txt'])

            # Create logging context
            self.epoch_train_log_writer = CSVLogWriter(
                os.path.join(self.log_dir, 'epoch_train_log.csv'), self.field_names)
            self.epoch_train_log_writer.open()

            self.iter_train_log_writer = CSVLogWriter(
                os.path.join(self.log_dir, 'iter_train_log.csv'), self.field_names)
            self.iter_train_log_writer.open()

            self.iter_val_log_writer = CSVLogWriter(
                os.path.join(self.log_dir, 'iter_val_log.csv'), self.field_names)
            self.iter_val_log_writer.open()

            # Checkpointer
            self.checkpointer = Checkpointer(
                model=self.model, optimiser=self.optimiser, scheduler=self.scheduler,
                save_dir=os.path.join(self.log_dir, 'checkpoints'),
                iter_period=self.iter_save_period, epoch_period=self.epoch_save_period)

            # Validator
            self.validator = EpochValidator(
                dataloader=self.val_dataloader, model=self.model, 
                save_dir=os.path.join(self.log_dir, 'val_logs'), 
                log_writer=self.iter_val_log_writer,
                epoch_period=self.epoch_val_period)

        # Iter counter
        self.iter_counter = Counter()

    def end_context(self):
        global rank
        if rank == 0:
            self.epoch_train_log_writer.close()
            self.iter_train_log_writer.close()
            self.iter_val_log_writer.close()

        # Reset 
        self.epoch_train_log_writer = None
        self.iter_train_log_writer = None
        self.iter_val_log_writer = None

        self.checkpointer = None
        self.validator = None
        self.iter_counter = None


@contextlib.contextmanager
def train_wrapper(
        log_dir: str,
        configs: Dict[str, Any],
        field_names: List[str],
        model: nn.Module,
        optimiser: optim.Optimizer,
        scheduler: optim.lr_scheduler.LRScheduler,
        train_dataloader: torch_data.DataLoader,
        val_dataloader: torch_data.DataLoader,
        iter_save_period: int,
        epoch_val_period: int,
        epoch_save_period: int
) -> Generator[TrainContext, None, None]:

    train_context = TrainContext(
        log_dir=log_dir, configs=configs, field_names=field_names,
        model=model, optimiser=optimiser, scheduler=scheduler,
        train_dataloader=train_dataloader, val_dataloader=val_dataloader,
        iter_save_period=iter_save_period, epoch_val_period=epoch_val_period,
        epoch_save_period=epoch_save_period)

    try:
        train_context.start_context()
        yield train_context
    finally:
        train_context.end_context()


@contextlib.contextmanager
def epoch_wrapper(epoch: int, train_context: TrainContext) -> Generator[AverageLogs, None, None]:
    global rank

    epoch_logs = AverageLogs()
    yield epoch_logs

    if rank == 0:
        train_context.checkpointer.save_epoch(epoch_counter=epoch)
        train_context.epoch_train_log_writer.write_log(epoch_logs.get_logs())
        train_context.validator.validate(epoch)


@contextlib.contextmanager
def iter_wrapper(train_context: TrainContext) -> Generator[InstantLogs, None, None]:
    global rank
    instant_logs = InstantLogs()
    yield instant_logs

    if rank == 0:
        train_context.checkpointer.save_iter(train_context.iter_counter.get_iter_counter())
        train_context.iter_counter.update()
        train_context.iter_train_log_writer.write_log(instant_logs.get_logs())


def train(
        identifier: str,
        train_configs: TrainConfigs,
        model_configs: SpeechEnhancementVAE.ConstructorArgs,
        pretrain_model_configs: PretrainSpeechVarianceVAE.ConstructorArgs,
        pretrain_model_path: str,
        init_state_dict_path: Optional[str] = None) -> None:

    global num_gpus
    global rank
    global group_name
    if num_gpus > 1:
        init_distributed(rank=rank, num_gpus=num_gpus, group_name=group_name, dist_backend='nccl', dist_url='tcp://localhost:54321')

    if train_configs.train_data_type == 'PreSynctheticNoisyDataset':
        train_dataset = PreSyntheticNoisyDataset(**train_configs.train_data.dict())
    elif train_configs.train_data_type == 'AugmentedNoisyDataset':
        train_dataset = AugmentedNoisyDataset(**train_configs.train_data.dict())
    val_dataset = PreSyntheticNoisyDataset(**train_configs.validation_data.dict())

    train_sampler = torch_data.DistributedSampler(
        train_dataset, shuffle=train_configs.train_dataloader.shuffle
    ) if num_gpus > 1 else None

    train_dataloader_configs = train_configs.train_dataloader.copy()
    train_dataloader_configs.batch_size //= num_gpus
    if train_sampler is not None:
        train_dataloader_configs.shuffle = False 

    train_dataloader = torch_data.DataLoader(
        dataset=train_dataset, sampler=train_sampler, **train_dataloader_configs.dict())
    val_dataloader = torch_data.DataLoader(
        dataset=val_dataset, **train_configs.validation_dataloader.dict())

    if init_state_dict_path is not None:
        (model_state_dict,
         optimiser_state_dict,
         scheduler_state_dict,
         previous_epoch) = load_state_dict_from_path(init_state_dict_path)

        model, optimiser, scheduler = load_model_to_cuda(
            model_class=SpeechEnhancementVAE,
            model_configs=model_configs.dict(),
            model_params_filters=lambda net: net.parameters(),
            optimiser_class=optim.Adam,
            optimiser_configs=train_configs.optimiser,
            scheduler_class=optim.lr_scheduler.OneCycleLR,
            scheduler_configs=train_configs.scheduler,
            model_state_dict=model_state_dict,
            optimiser_state_dict=optimiser_state_dict,
            scheduler_state_dict=scheduler_state_dict)
    else:
        pretrain_model_state_dict, _, _, _ = load_state_dict_from_path(pretrain_model_path)
        optimiser_state_dict, scheduler_state_dict, previous_epoch = None, None, -1

        model, optimiser, scheduler = load_model_to_cuda(
            model_class=SpeechEnhancementVAE,
            model_configs=model_configs.dict(),
            model_params_filters=lambda net: net.parameters(),
            model_load_state_dict_method_name='load_state_dict_from_pretrain_state_dict',
            optimiser_class=optim.Adam,
            optimiser_configs=train_configs.optimiser,
            scheduler_class=optim.lr_scheduler.OneCycleLR,
            scheduler_configs=train_configs.scheduler,
            model_state_dict={
                'configs': pretrain_model_configs.dict(),
                'state_dict': pretrain_model_state_dict},
            optimiser_state_dict=optimiser_state_dict,
            scheduler_state_dict=scheduler_state_dict)

    if train_configs.auto_mix_precision:
        scaler = torch.cuda.amp.GradScaler()

    log_dir = 'results/is2022/{}_{}'.format(
        identifier, datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    list_fields = model.output_logging_keys

    with train_wrapper(
            log_dir=log_dir,
            configs={'model': model_configs.dict(), 'train': train_configs.dict()},
            field_names=list_fields,
            model=model,
            optimiser=optimiser,
            scheduler=scheduler,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            iter_save_period=train_configs.iter_per_checkpoint,
            epoch_val_period=train_configs.epoch_per_val,
            epoch_save_period=train_configs.epoch_per_checkpoint
    ) as train_context:
        for epoch in range(previous_epoch + 1, train_configs.n_epochs):
            with epoch_wrapper(epoch, train_context) as epoch_logs:
                if train_sampler is not None:
                    train_sampler.set_epoch(epoch)

                for batch_data in train_dataloader:
                    y_bt = batch_data[0]
                    x_bt = batch_data[1]

                    with iter_wrapper(train_context) as iter_logs:
                        model.train()

                        model.zero_grad()
                        optimiser.zero_grad()

                        if train_configs.auto_mix_precision:
                            # Train with GradScaler
                            with torch.cuda.amp.autocast(enabled=True):
                                x_bt = x_bt.cuda()
                                y_bt = y_bt.cuda()
                                loss, logs = model(x_bt, y_bt, epoch=epoch)

                            scaler.scale(loss).backward()
                            scaler.step(optimiser)
                            scale_value = scaler.get_scale()
                            scaler.update()
                            skip_scheduler = scale_value > scaler.get_scale()
                            if not skip_scheduler and scheduler is not None:
                                scheduler.step()
                        else:
                            # Train without GradScaler
                            x_bt = x_bt.cuda()
                            y_bt = y_bt.cuda()
                            loss, logs = model(x_bt, y_bt, epoch=epoch)
                            loss.backward()
                            optimiser.step()
                            if scheduler is not None:
                                scheduler.step()

                        iter_logs.update(logs)
                        epoch_logs.update(logs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--identifier', type=str, required=True)
    parser.add_argument('--train-configs', type=str, required=True)
    parser.add_argument('--model-configs', type=str, required=True)
    parser.add_argument('--pretrain-model-configs', type=str, required=True)
    parser.add_argument('--pretrain-model-path', type=str, required=True)
    parser.add_argument('--init-state-dict-path', type=str, required=False, default=None)
    parser.add_argument('--rank', type=int, required=False, default=0)
    parser.add_argument('--group-name', type=str, required=False, default='')

    args = parser.parse_args()
    with open(args.train_configs, 'r') as f:
        train_configs = yaml.safe_load(f)
    with open(args.model_configs, 'r') as f:
        model_configs = yaml.safe_load(f)
    with open(args.pretrain_model_configs, 'r') as f:
        pretrain_model_configs = yaml.safe_load(f)
    train_configs = TrainConfigs(**train_configs)
    model_configs = SpeechEnhancementVAE.ConstructorArgs(**model_configs)
    pretrain_model_configs = PretrainSpeechVarianceVAE.ConstructorArgs(**pretrain_model_configs)
    identifier = args.identifier

    global num_gpus
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        if args.group_name == '':
            print('Warning: Training on 1 GPU!')
            num_gpus = 1
        else:
            print(f'Run distributed training on {num_gpus} GPUs')

    global rank
    rank = args.rank

    global group_name
    group_name = args.group_name

    train(identifier=identifier, train_configs=train_configs, model_configs=model_configs,
          pretrain_model_configs=pretrain_model_configs,
          pretrain_model_path=args.pretrain_model_path,
          init_state_dict_path=args.init_state_dict_path)


if __name__ == '__main__':
    main()
