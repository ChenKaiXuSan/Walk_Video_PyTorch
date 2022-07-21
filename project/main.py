# %%
import os
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import loggers as pl_loggers
# callbacks
from pytorch_lightning.callbacks import TQDMProgressBar, RichModelSummary, RichProgressBar
from pl_bolts.callbacks import PrintTableMetricsCallback, TrainingDataMonitor
from utils.utils import get_ckpt_path

from dataloader.data_loader import WalkDataModule
from models.pytorchvideo_models import WalkVideoClassificationLightningModule
from argparse import ArgumentParser

import pytorch_lightning
# %%

def get_parameters():

    parser = ArgumentParser()

    # model hyper-parameters
    parser.add_argument('--model', type=str, default='resnet', choices=['resnet', 'csn'])
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--version', type=str, default='test', help='the version of logger, such data')
    parser.add_argument('--model_class_num', type=int, default=2, help='the class num of model')
    parser.add_argument('--model_depth', type=int, default=50, choices=[50, 101, 152], help='the depth of used model')

    # Training setting
    parser.add_argument('--max_epochs', type=int, default=100, help='numer of epochs of training')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size for the dataloader')
    parser.add_argument('--num_workers', type=int, default=2, help='dataloader for load video')
    parser.add_argument('--clip_duration', type=int, default=2, help='clip duration for the video')
    parser.add_argument('--gpu_num', type=int, default=0, choices=[0, 1], help='the gpu number whicht to train')

    # TTUR
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate for optimizer')
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)

    # Path
    parser.add_argument('--data_path', type=str, default="/workspace/data/walk_data_finish_train/lat/", help='meta dataset path')
    parser.add_argument('--split_data_path', type=str, default="/workspace/data/dataset/", help="split dataset path")

    parser.add_argument('--log_path', type=str, default='./logs', help='the lightning logs saved path')

    # using pretrained 
    parser.add_argument('--pretrained_model', type=bool, default=False, help='if use the pretrained model for training.')
    # add the parser to ther Trainer 
    # parser = Trainer.add_argparse_args(parser)

    return parser.parse_known_args()


# %%

def train(hparams):

    # connect the version + model + depth
    hparams.version = hparams.version + '_' + hparams.model + '_depth' + str(hparams.depth)

    # fixme will occure bug, with deterministic = true
    # seed_everything(42, workers=True)

    classification_module = WalkVideoClassificationLightningModule(hparams)

    # instance the data module
    data_module = WalkDataModule(hparams)

    # for the tensorboard
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=hparams.log_path, name=hparams.model, version=hparams.version)

    # some callbacks
    progress_bar = TQDMProgressBar(refresh_rate=hparams.batch_size)
    rich_model_summary = RichModelSummary(max_depth=5)
    rich_progress_bar = RichProgressBar(refresh_rate=hparams.batch_size)

    # bolts callbacks
    table_metrics_callback = PrintTableMetricsCallback()
    monitor = TrainingDataMonitor(log_every_n_steps=50)

    trainer = Trainer(accelerator="auto",
                      devices=1, 
                      gpus=hparams.gpu_num,
                      max_epochs=hparams.max_epochs,
                      logger=tb_logger,
                      log_every_n_steps=100,
                      check_val_every_n_epoch=10,
                      callbacks=[progress_bar, rich_model_summary, table_metrics_callback, monitor],
                    #   deterministic=True 
                      )

    # from the params
    # trainer = Trainer.from_argparse_args(hparams)
    
    if hparams.pretrained_model:
        trainer.fit(classification_module, data_module, ckpt_path=get_ckpt_path)
    else:    
        # training and val
        trainer.fit(classification_module, data_module)

    # testing
    # trainer.test(dataloaders=data_module)

    # predict 
    # trainer.predict(dataloaders=data_module)


# %%
if __name__ == '__main__':

    # for test in jupyter 
    config, unkonwn = get_parameters()

    train(config)
