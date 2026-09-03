import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from model import *
from data import PatchSet
from utils import *
from timeit import default_timer as timer
from datetime import datetime
import numpy as np
import pandas as pd
import shutil


class Experiment(object):
    def __init__(self, option):
        self.device = torch.device('cuda' if option.cuda else 'cpu')
        self.image_size = make_tuple(option.image_size)
        self.save_dir = option.save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.train_dir = self.save_dir / 'train'
        self.train_dir.mkdir(exist_ok=True)
        self.history = self.train_dir / 'history.csv'
        self.test_dir = self.save_dir / 'test'
        self.test_dir.mkdir(exist_ok=True)
        self.checkpoint = self.train_dir / 'last.pth'
        self.best = self.train_dir / 'best.pth'
        self.logger=setup_logger(option.logpath)
        self.logger.info('Model initialization')

        # 模型初始化：
        self.model = FusionNet_MWM().to(self.device)


        if option.cuda and option.ngpu > 1:
            device_ids = [i for i in range(option.ngpu)]
            self.model = nn.DataParallel(self.model, device_ids=device_ids)


        self.criterion = CompoundLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001, weight_decay=1e-6)

        self.logger.info(str(self.model))
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f'There are {n_params} trainable parameters.')


    def train_on_epoch(self, n_epoch, data_loader):
            self.model.train()
            epoch_loss = AverageMeter()
            epoch_score = AverageMeter()
            batches = len(data_loader)
            self.logger.info(f'Epoch[{n_epoch}] - {datetime.now()}')

            for idx, data in enumerate(data_loader):
                t_start = timer()
                data = [im.to(self.device) for im in data]
                inputs, target = data[:-1], data[-1]

                self.optimizer.zero_grad()
                predictions = self.model(inputs)
                loss = self.criterion(predictions, target)
                epoch_loss.update(loss.item())
                loss.backward()
                self.optimizer.step()

                with torch.no_grad():
                    score = F.mse_loss(predictions, target)
                epoch_score.update(score.item())

                t_end = timer()
                self.logger.info(f'Epoch[{n_epoch} {idx}/{batches}] - '
                                 f'Loss: {loss.item():.10f} - '
                                 f'MSE: {score.item():.5f} - '
                                 f'Time: {t_end - t_start}s')
            return epoch_loss.avg, epoch_score.avg

    @torch.no_grad()
    def test_on_epoch(self, data_loader):
        self.model.eval()
        epoch_loss = AverageMeter()
        epoch_error = AverageMeter()

        for data in data_loader:
            data = [im.to(self.device) for im in data]
            inputs = data[:-1]
            target = data[-1]
            prediction = self.model(inputs)
            loss = self.criterion(prediction, target)
            epoch_loss.update(loss.item())
            score = F.mse_loss(prediction, target)
            epoch_error.update(score.item())

        save_checkpoint(self.model, self.optimizer, self.checkpoint)
        return epoch_loss.avg, epoch_error.avg


    def train(self, train_dir, val_dir, patch_size, patch_stride, batch_size,
              num_workers=0, epochs=200, resume=True):
        self.logger.info('Loading data...')
        train_set = PatchSet(train_dir, self.image_size, patch_size, patch_stride)
        val_set = PatchSet(val_dir, self.image_size, patch_size)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=num_workers)

        least_error = 1.0
        start_epoch = 0
        if resume and self.checkpoint.exists():
            load_checkpoint(self.checkpoint, model=self.model, optimizer=self.optimizer)

            if self.history.exists():
                df = pd.read_csv(self.history)
                least_error = df['val_error'].min()
                start_epoch = int(df.iloc[-1]['epoch']) + 1

        self.logger.info('Training...')
        for epoch in range(start_epoch, epochs + start_epoch):
            for param_group in self.optimizer.param_groups:
                self.logger.info(f"Current learning rate: {param_group['lr']}")

            train_loss, train_error = self.train_on_epoch(epoch, train_loader)
            val_loss, val_error = self.test_on_epoch(val_loader)
            csv_header = ['epoch', 'train_loss', 'train_error', 'val_loss', 'val_error']
            csv_values = [epoch, train_loss, train_error, val_loss, val_error]
            log_csv(self.history, csv_values, header=csv_header)

            if val_error <= least_error:
                shutil.copy(self.checkpoint, self.best)
                least_error = val_error

            epoch_save_path = self.train_dir / f'epoch_{epoch}.pth'
            save_checkpoint(self.model, self.optimizer, epoch_save_path)


    @torch.no_grad()
    def test(self, test_dir, num_workers=0):
        self.model.eval()
        load_checkpoint(self.best, model=self.model)
        self.logger.info('Testing...')
        image_dirs = iter([p for p in test_dir.iterdir() if p.is_dir()])
        test_set = PatchSet(test_dir, self.image_size, self.image_size)
        test_loader = DataLoader(test_set, batch_size=1, num_workers=num_workers)

        pixel_value_scale = 255

        for inputs in test_loader:
            t_start = timer()
            inputs = [im.to(self.device) for im in inputs[:-1]]
            name = f'PRED_{next(image_dirs).stem}.tif'
            prediction = self.model(inputs)
            prediction = prediction.squeeze().cpu().numpy()
            result = (prediction * pixel_value_scale).astype(np.int16)

            # 存储预测影像结果
            metadata = {
                'driver': 'GTiff',
                'width': self.image_size[1],
                'height': self.image_size[0],
                'count': NUM_BANDS,
                'dtype': np.int16
            }
            save_array_as_tif(result, self.test_dir / name, metadata)

            del inputs, prediction, result
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            t_end = timer()
            self.logger.info(f'Time cost: {t_end - t_start}s')
