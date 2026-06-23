import os, sys
import time
import yaml
import platform
import torch
from argparse import ArgumentParser
import lightning as L
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from utils.utils_training import CustomProgressBar, SaveBestStateDictCallback
from utils_model.CVAE import CVAE
from utils_model.PointNetLabels import PointNetLabels
from utils_data.DataModule import DataModule
from utils.constants import ROOT_PATH


def get_parser():
    parser = ArgumentParser()
    parser.add_argument('--train_name', default='', type=str)
    parser.add_argument('--model', default='cvae', type=str, choices=['cvae', 'pointnet'], help='Model to train')
    parser.add_argument('--robot_name', default='allegro', type=str, help='Name of the robot')
    parser.add_argument('--resume', default=False, action='store_true', help='Resume training from last checkpoint')
    args_ = parser.parse_args()
    return args_


if __name__ == "__main__":
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision('high')
    # torch.autograd.set_detect_anomaly(True)

    args = get_parser()

    if args.resume:
        train_name = args.robot_name + '_' + args.model + '_' + args.train_name
        log_dir = os.path.join('logs', train_name)
        chkpt_dir = os.path.join(log_dir, 'ckpts_dir')

        # 0 - Load config
        config_path = os.path.join(log_dir, 'config.yaml')
        with open(config_path, "r") as file:
            cfg = yaml.safe_load(file)
        cfg = cfg[args.model]
        cfg['robot_name'] = args.robot_name
        
    else:
        # 0 - Load config
        config_path = os.path.join(ROOT_PATH, 'configs/train.yaml')
        with open(config_path, "r") as file:
            cfg = yaml.safe_load(file)

        # 1 - Prepare logs basedir
        cfg = cfg[args.model]
        cfg['robot_name'] = args.robot_name
        train_name = cfg['robot_name'] + '_' + args.model + '_' + args.train_name

        log_dir = os.path.join('logs', train_name)
        chkpt_dir = os.path.join(log_dir, 'ckpts_dir')

        if os.path.exists(log_dir):
            if not 'node' in platform.node():
                print(f"Log directory {log_dir} already exists. Removing it ? (y/n)", end=' ')
                response = input().strip().lower()
                if response == 'y':
                    os.system(f'rm -rf "{log_dir}"')
                else:
                    sys.exit()
            else:
                os.system(f'rm -rf "{log_dir}"')
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(chkpt_dir, exist_ok=True)

        # os.system(f'cp {config_path} {os.path.join(log_dir, "config.yaml")}')
        with open(os.path.join(log_dir, 'config.yaml'), "w") as file:
            yaml.dump(cfg, file)

    # Tensorboard logger
    tb_logger = TensorBoardLogger(save_dir=log_dir, name='tb_dir', log_graph=False)

    checkpoint_callback = ModelCheckpoint(
        dirpath=chkpt_dir,
        filename=f'{train_name}' + '-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        save_last=True,
        monitor='val_loss',
        mode='min'
    )

    print(f'Training {args.model.upper()} on: {platform.node()}')


    # 1 - DataModule
    data_module = DataModule(cfg, shuffle=False)


    # 2 - Init Model
    print("Init model... ", end='\r')
    if args.model == 'cvae':
        model = CVAE(cfg)
    elif args.model == 'pointnet':
        model = PointNetLabels(cfg)

    model.train()
    print("Init model... OK")


    # 3 - Train
    trainer = L.Trainer(
        accelerator='gpu',
        devices='auto',
        max_epochs=cfg['n_epochs'],
        logger=tb_logger,
        callbacks=[CustomProgressBar(), checkpoint_callback, SaveBestStateDictCallback(chkpt_dir, train_name=train_name, save_best_epoch=False)],
        default_root_dir=log_dir,
        log_every_n_steps=1,
    )

    start_time = time.time()

    if args.resume:
        # Load the last checkpoint
        last_checkpoint = os.path.join(chkpt_dir, 'last.ckpt')
        trainer.fit(model, datamodule=data_module, ckpt_path=last_checkpoint)
    else:
        trainer.fit(model, datamodule=data_module)
    trainer.save_checkpoint(os.path.join(chkpt_dir, f'{train_name}.ckpt'))

    print(f"Done with training {train_name} on {cfg['n_epochs']} epochs with a batchsize of {cfg['batch_size']}.")
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Total time: {int(hours):02}:{int(minutes):02}:{int(seconds):02}")